import tensorflow as tf
from tf_gen_models.algorithms.gan import GAN


class WGAN (GAN):
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
                clip_param = 0.01 ) -> None:
    super().compile ( g_optimizer = g_optimizer , 
                      d_optimizer = d_optimizer , 
                      g_updt_per_batch = g_updt_per_batch , 
                      d_updt_per_batch = d_updt_per_batch )

    ## data-type control
    if not isinstance (clip_param, float):
      if isinstance (clip_param, int): float (clip_param)
      else: raise TypeError ("The clipping parameter should be a float.")

    ## data-value control
    if clip_param <= 0:
      raise ValueError ("The clipping parameter should be greater than 0.")

    self._clip_param = clip_param

  @tf.function
  def _train_d_step (self, X, Y, w = None) -> None:
    super()._train_d_step ( X = X, Y = Y, w = w )
    for w in self._discriminator.trainable_weights:
      w = tf.clip_by_value ( w, - self._clip_param, self._clip_param )   # weights clipping

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
    """The discriminator of the WGAN system."""
    return self._discriminator

  @property
  def generator (self) -> tf.keras.Sequential:
    """The generator of the WGAN system."""
    return self._generator

  @property
  def clip_param (self) -> float:
    """Clipping parameter for the discriminator weights."""
    return self._clip_param
