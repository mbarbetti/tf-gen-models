import tensorflow as tf
from tf_gen_models.algorithms.gan import GAN


class BceGAN (GAN):
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
    self._loss_name = "Binary cross entropy"

  def compile ( self , 
                g_optimizer , 
                d_optimizer , 
                g_updt_per_batch = 1 , 
                d_updt_per_batch = 1 ) -> None:
    super().compile ( g_optimizer = g_optimizer , 
                      d_optimizer = d_optimizer , 
                      g_updt_per_batch = g_updt_per_batch , 
                      d_updt_per_batch = d_updt_per_batch )

    self._k_gen = 0.1
    self._k_ref = 0.9

  def _compute_g_loss (self, gen_sample, ref_sample) -> tf.Tensor:
    ## extract input tensors and weights
    input_gen, w_gen = gen_sample
    input_ref, w_ref = ref_sample

    ## noise injection to stabilize BceGAN training
    rnd_gen = tf.random.normal ( tf.shape (input_gen), stddev = 0.05, dtype = input_gen.dtype )
    rnd_ref = tf.random.normal ( tf.shape (input_ref), stddev = 0.05, dtype = input_ref.dtype )
    D_gen = tf.cast ( self._discriminator (input_gen + rnd_gen), dtype = input_gen.dtype )
    D_ref = tf.cast ( self._discriminator (input_ref + rnd_ref), dtype = input_ref.dtype )

    ## loss computation
    g_loss = w_gen * self._k_gen       * tf.math.log ( tf.clip_by_value ( D_gen     , 1e-12 , 1. ) ) + \
             w_gen * (1 - self._k_gen) * tf.math.log ( tf.clip_by_value ( 1 - D_gen , 1e-12 , 1. ) ) + \
             w_ref * self._k_ref       * tf.math.log ( tf.clip_by_value ( D_ref     , 1e-12 , 1. ) ) + \
             w_ref * (1 - self._k_ref) * tf.math.log ( tf.clip_by_value ( 1 - D_ref , 1e-12 , 1. ) ) 
    return tf.reduce_mean (g_loss, axis = None)

  def _compute_threshold (self, ref_sample) -> tf.Tensor:
    ## extract input tensors and weights
    input_ref, w_ref = ref_sample

    ## noise injection to stabilize GAN training
    rnd_ref = tf.random.normal ( tf.shape (input_ref), stddev = 0.05, dtype = input_ref.dtype )
    D_ref = tf.cast ( self._discriminator (input_ref + rnd_ref), dtype = input_ref.dtype )

    ## split tensors and weights
    batch_size = tf.cast ( tf.shape(input_ref)[0] / 2, tf.int32 )
    D_ref_1, D_ref_2 = D_ref[:batch_size], D_ref[batch_size:batch_size*2]
    w_ref_1, w_ref_2 = w_ref[:batch_size], w_ref[batch_size:batch_size*2]

    ## threshold loss computation
    th_loss = w_ref_1 * self._k_gen       * tf.math.log ( tf.clip_by_value ( D_ref_1     , 1e-12 , 1. ) ) + \
              w_ref_1 * (1 - self._k_gen) * tf.math.log ( tf.clip_by_value ( 1 - D_ref_1 , 1e-12 , 1. ) ) + \
              w_ref_2 * self._k_ref       * tf.math.log ( tf.clip_by_value ( D_ref_2     , 1e-12 , 1. ) ) + \
              w_ref_2 * (1 - self._k_ref) * tf.math.log ( tf.clip_by_value ( 1 - D_ref_2 , 1e-12 , 1. ) ) 
    return tf.reduce_mean (th_loss, axis = None)

  @property
  def discriminator (self) -> tf.keras.Sequential:
    """The discriminator of the BceGAN system."""
    return self._discriminator

  @property
  def generator (self) -> tf.keras.Sequential:
    """The generator of the BceGAN system."""
    return self._generator

  @property
  def k_gen (self) -> float:
    """Smoothness weight of the BCE for the reference dataset."""
    return self._k_gen

  @k_gen.setter
  def k_gen (self, k) -> None:
    ## data-type control
    if not isinstance (k_gen, float):
      if isinstance (k_gen, int): k_gen = float (k_gen)
      else: raise TypeError ("The smoothness weight should be a float.")

    ## data-value control
    if (k_gen < 0.0) or (k_gen > 1.0):
      raise ValueError ("The smoothness weight should be between 0 and 1.")

    self._k_gen = k

  @property
  def k_ref (self) -> float:
    """Smoothness weight of the BCE for the generated dataset."""
    return self._k_ref

  @k_ref.setter
  def k_ref (self, k) -> None:
    ## data-type control
    if not isinstance (k_ref, float):
      if isinstance (k_ref, int): k_ref = float (k_ref)
      else: raise TypeError ("The smoothness weight should be a float.")

    ## data-value control
    if (k_ref < 0.0) or (k_ref > 1.0):
      raise ValueError ("The smoothness weight should be between 0 and 1.")

    self._k_ref = k
