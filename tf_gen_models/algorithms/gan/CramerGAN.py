import tensorflow as tf
from tf_gen_models.algorithms.gan import GAN


class Critic:
  """Critic function."""
  def __init__ (self, h):
    self.h = h

  def __call__ (self, x_1, x_2):
    critic_func = tf.norm (self.h(x_1) - self.h(x_2), axis = 1) - tf.norm (self.h(x_1), axis = 1)
    return critic_func


class CramerGAN (GAN):
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
    self._loss_name = "Energy distance"

  def compile ( self , 
                g_optimizer , 
                d_optimizer ,
                g_updt_per_batch = 1 , 
                d_updt_per_batch = 1 ,
                grad_penalty = 1.0 ) -> None:
    super().compile ( g_optimizer = g_optimizer , 
                      d_optimizer = d_optimizer , 
                      g_updt_per_batch = g_updt_per_batch , 
                      d_updt_per_batch = d_updt_per_batch )

    self._critic = Critic ( lambda x : self._discriminator(x) )

    ## data-type control
    if not isinstance (grad_penalty, float):
      raise TypeError ("The loss gradient penalty should be a float.")

    ## data-value control
    if grad_penalty <= 0:
      raise ValueError ("The loss gradient penalty should be greater than 0.")

    self._grad_penalty = grad_penalty
  
  def _compute_d_loss (self, gen_sample, ref_sample) -> tf.Tensor:
    ## extract input tensors and weights
    feats_gen, w_gen = gen_sample
    feats_ref, w_ref = ref_sample

    ## data-batch splitting
    batch_size = tf.cast ( tf.shape(feats_gen)[0] / 2, tf.int32 )

    feats_gen_1, feats_gen_2 = feats_gen[:batch_size], feats_gen[batch_size:batch_size*2]
    w_gen_1, w_gen_2 = w_gen[:batch_size], w_gen[batch_size:batch_size*2]

    feats_ref_1 = feats_ref[:batch_size]
    w_ref_1 = w_ref[:batch_size]

    ## discriminator loss computation
    d_loss = w_gen_1 * w_gen_2 * self._critic ( feats_gen_1, feats_gen_2 ) - \
             w_ref_1 * w_gen_2 * self._critic ( feats_ref_1, feats_gen_2 )
    d_loss = tf.reduce_mean (d_loss)

    ## data interpolation
    alpha = tf.random.uniform (
                                shape  = (tf.shape(feats_ref_1)[0],) ,
                                minval = 0.0 ,
                                maxval = 1.0 ,
                                dtype  = feats_ref_1.dtype
                              )
    one_shape = tf.ones_like ( tf.shape(feats_ref)[1:] )
    new_shape = tf.concat ( [tf.shape(alpha), one_shape], axis = 0 )
    alpha = tf.reshape ( alpha, shape = new_shape )
    differences  = feats_gen_1 - feats_ref_1
    interpolates = feats_ref_1 + alpha * differences

    ## gradient penalty correction
    critic_int = self._critic ( interpolates, feats_gen_2 )
    grad = tf.gradients ( critic_int , interpolates )
    grad = tf.concat  ( grad , axis = 1 )
    grad = tf.reshape ( grad , shape = (tf.shape(grad)[0], -1) )
    slopes  = tf.norm ( grad , axis = 1 )
    gp_term = tf.square ( tf.maximum ( tf.abs (slopes) - 1.0, 0.0 ) )
    gp_term = self._grad_penalty * tf.reduce_mean (gp_term)   # gradient penalty
    d_loss += gp_term
    return d_loss

  def _compute_g_loss (self, gen_sample, ref_sample) -> tf.Tensor:
    ## extract input tensors and weights
    feats_gen, w_gen = gen_sample
    feats_ref, w_ref = ref_sample

    ## data-batch splitting
    batch_size = tf.cast ( tf.shape(feats_gen)[0] / 2, tf.int32 )

    feats_gen_1, feats_gen_2 = feats_gen[:batch_size], feats_gen[batch_size:batch_size*2]
    w_gen_1, w_gen_2 = w_gen[:batch_size], w_gen[batch_size:batch_size*2]

    feats_ref_1 = feats_ref[:batch_size]
    w_ref_1 = w_ref[:batch_size]

    ## generator loss computation
    g_loss = w_ref_1 * w_gen_2 * self._critic ( feats_ref_1, feats_gen_2 ) - \
             w_gen_1 * w_gen_2 * self._critic ( feats_gen_1, feats_gen_2 )
    return tf.reduce_mean (g_loss)

  def _compute_threshold (self, ref_sample) -> tf.Tensor:
    _ , w_ref = ref_sample
    th_loss = tf.zeros_like (w_ref)
    return tf.reduce_mean (th_loss)

  @property
  def discriminator (self) -> tf.keras.Sequential:
    """The discriminator of the CramerGAN system."""
    return self._discriminator

  @property
  def generator (self) -> tf.keras.Sequential:
    """The generator of the CramerGAN system."""
    return self._generator

  @property
  def critic_dim (self) -> int:
    """The dimension of the critic space."""
    return self._discriminator.output_shape[1]
