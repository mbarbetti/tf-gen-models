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

  def _compute_g_loss (self, gen_sample, ref_sample) -> tf.Tensor:
    ## extract input tensors and weights
    feats_gen, w_gen = gen_sample
    feats_ref, w_ref = ref_sample

    ## noise injection to stabilize BceGAN training
    rnd_gen = tf.random.normal ( tf.shape(feats_gen), stddev = 0.05, dtype = feats_gen.dtype )
    rnd_ref = tf.random.normal ( tf.shape(feats_ref), stddev = 0.05, dtype = feats_ref.dtype )
    D_gen = tf.cast ( self._discriminator ( feats_gen + rnd_gen ), dtype = feats_gen.dtype )
    D_ref = tf.cast ( self._discriminator ( feats_ref + rnd_ref ), dtype = feats_ref.dtype )

    ## loss computation
    true_gen = 0.9
    true_ref = 0.1
    g_loss = w_gen * true_gen       * tf.math.log ( tf.clip_by_value ( D_gen     , 1e-12 , 1. ) ) + \
             w_gen * (1 - true_gen) * tf.math.log ( tf.clip_by_value ( 1 - D_gen , 1e-12 , 1. ) ) + \
             w_ref * true_ref       * tf.math.log ( tf.clip_by_value ( D_ref     , 1e-12 , 1. ) ) + \
             w_ref * (1 - true_ref) * tf.math.log ( tf.clip_by_value ( 1 - D_ref , 1e-12 , 1. ) ) 
    return tf.reduce_mean (g_loss)

  def _compute_threshold (self, ref_sample) -> tf.Tensor:
    ## extract input tensors and weights
    feats_ref, w_ref = ref_sample

    ## noise injection to stabilize GAN training
    rnd_ref = tf.random.normal ( tf.shape(feats_ref), stddev = 0.05, dtype = feats_ref.dtype )
    D_ref = tf.cast ( self._discriminator ( feats_ref + rnd_ref ), dtype = feats_ref.dtype )

    ## split tensors and weights
    batch_size = tf.cast ( tf.shape(feats_ref)[0] / 2, tf.int32 )
    D_ref_1, D_ref_2 = D_ref[:batch_size], D_ref[batch_size:batch_size*2]
    w_ref_1, w_ref_2 = w_ref[:batch_size], w_ref[batch_size:batch_size*2]

    ## threshold loss computation
    true_gen = 0.9
    true_ref = 0.1
    th_loss = w_ref_1 * true_gen       * tf.math.log ( tf.clip_by_value ( D_ref_1     , 1e-12 , 1. ) ) + \
              w_ref_1 * (1 - true_gen) * tf.math.log ( tf.clip_by_value ( 1 - D_ref_1 , 1e-12 , 1. ) ) + \
              w_ref_2 * true_ref       * tf.math.log ( tf.clip_by_value ( D_ref_2     , 1e-12 , 1. ) ) + \
              w_ref_2 * (1 - true_ref) * tf.math.log ( tf.clip_by_value ( 1 - D_ref_2 , 1e-12 , 1. ) ) 
    return tf.reduce_mean (th_loss)

  @property
  def discriminator (self) -> tf.keras.Sequential:
    """The discriminator of the BceGAN system."""
    return self._discriminator

  @property
  def generator (self) -> tf.keras.Sequential:
    """The generator of the BceGAN system."""
    return self._generator
