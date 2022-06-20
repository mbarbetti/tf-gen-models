import tensorflow as tf
from tf_gen_models.algorithms.gan import GAN


class WGAN_GP (GAN):
  def __init__ ( self , 
                 generator     ,
                 discriminator , 
                 input_shape  = None ,
                 output_shape = None ,
                 latent_dim = 128 ) -> None:
    super().__init__ ( generator = generator ,
                       discriminator = discriminator , 
                       input_shape = input_shape ,
                       output_shape = output_shape ,
                       latent_dim = latent_dim )
    self._loss_name = "Wasserstein distance"

  def compile ( self , 
                g_optimizer , 
                d_optimizer ,
                g_updt_per_batch = 1 , 
                d_updt_per_batch = 1 ,
                grad_penalty = 10 ) -> None:
    super().compile ( g_optimizer = g_optimizer , 
                      d_optimizer = d_optimizer , 
                      g_updt_per_batch = g_updt_per_batch , 
                      d_updt_per_batch = d_updt_per_batch )

    ## data-type control
    if not isinstance (grad_penalty, float):
      if isinstance (grad_penalty, int): grad_penalty = float (grad_penalty)
      else: raise TypeError ("The loss gradient penalty should be a float.")

    ## data-value control
    if grad_penalty <= 0:
      raise ValueError ("The loss gradient penalty should be greater than 0.")

    self._grad_penalty = grad_penalty

  def _compute_d_loss (self, gen_sample, ref_sample) -> tf.Tensor:
    ## extract input tensors and weights
    input_gen, w_gen = gen_sample
    input_ref, w_ref = ref_sample

    ## standard WGAN loss
    D_gen = tf.cast ( self._discriminator (input_gen), dtype = input_gen.dtype )
    D_ref = tf.cast ( self._discriminator (input_ref), dtype = input_ref.dtype )
    d_loss = tf.reduce_mean ( w_gen * D_gen - w_ref * D_ref , axis = None )
    
    ## data interpolation
    alpha = tf.random.uniform ( shape  = (tf.shape(input_ref)[0],) ,
                                minval = 0.0 ,
                                maxval = 1.0 ,
                                dtype  = input_ref.dtype )
    new_shape = tf.ones_like ( tf.shape(input_ref)[1:] )
    new_shape = tf.concat ( [tf.shape(alpha), new_shape], axis = 0 )
    alpha = tf.reshape ( alpha, shape = new_shape )
    differences  = input_gen - input_ref
    interpolates = input_ref + alpha * differences

    ## gradient penalty correction
    D_int = self._discriminator ( interpolates )
    grad = tf.gradients ( D_int , [interpolates] ) [0]
    slopes  = tf.norm ( grad , axis = 1 )
    gp_term = tf.reduce_mean ( tf.square (slopes - 1.0) , axis = None )   # two-sided penalty
    d_loss += self._grad_penalty * gp_term   # gradient penalty
    return d_loss

  def _compute_g_loss (self, gen_sample, ref_sample) -> tf.Tensor:
    ## extract input tensors and weights
    input_gen, w_gen = gen_sample
    input_ref, w_ref = ref_sample

    ## standard WGAN loss
    D_gen = tf.cast ( self._discriminator (input_gen), dtype = input_gen.dtype )
    D_ref = tf.cast ( self._discriminator (input_ref), dtype = input_ref.dtype )
    g_loss = w_ref * D_ref - w_gen * D_gen
    return tf.reduce_mean (g_loss, axis = None)

  def _compute_threshold (self, ref_sample) -> tf.Tensor:
    _ , w_ref = ref_sample
    th_loss = tf.zeros_like (w_ref)
    return tf.reduce_sum (th_loss, axis = None)

  @property
  def discriminator (self) -> tf.keras.Sequential:
    """The discriminator of the WGAN-GP system."""
    return self._discriminator

  @property
  def generator (self) -> tf.keras.Sequential:
    """The generator of the WGAN-GP system."""
    return self._generator

  @property
  def grad_penalty (self) -> float:
    """Gradient penalty coefficient."""
    return self._grad_penalty
    