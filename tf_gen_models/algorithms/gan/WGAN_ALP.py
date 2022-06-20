import inspect
import tensorflow as tf
from tf_gen_models.algorithms.gan import GAN


class WGAN_ALP (GAN):
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
                v_adv_dir_updt   = 1 , 
                adv_lp_penalty = 100 ) -> None:
    super().compile ( g_optimizer = g_optimizer , 
                      d_optimizer = d_optimizer , 
                      g_updt_per_batch = g_updt_per_batch , 
                      d_updt_per_batch = d_updt_per_batch )

    ## data-type control
    if not isinstance (v_adv_dir_updt, int):
      if isinstance (v_adv_dir_updt, float): v_adv_dir_updt = int (v_adv_dir_updt)
      else: raise TypeError ("The number of virtual adversarial direction updates should be an integer.")

    if not isinstance (adv_lp_penalty, float):
      if isinstance (adv_lp_penalty, int): adv_lp_penalty = float (adv_lp_penalty)
      else: raise TypeError ("The adversarial Lipschitz penalty should be a float.")

    ## data-value control
    if v_adv_dir_updt <= 0:
      raise ValueError ("The number of virtual adversarial direction updates should be greater than 0.")

    if adv_lp_penalty <= 0:
      raise ValueError ("The adversarial Lipschitz penalty should be greater than 0.")

    self._v_adv_dir_updt = v_adv_dir_updt
    self._adv_lp_penalty = adv_lp_penalty
    self._lp_const = 1.0
    self._epsilon_sampler = lambda shape, dtype: tf.math.exp ( tf.random.uniform ( shape  = shape ,
                                                                                   minval = tf.math.log (1e-1) ,
                                                                                   maxval = tf.math.log (1e+1) ,
                                                                                   dtype  = dtype ) )
    self._xi = 10.0

  def _compute_d_loss (self, gen_sample, ref_sample) -> tf.Tensor:
    ## extract input tensors and weights
    input_gen, w_gen = gen_sample
    input_ref, w_ref = ref_sample

    ## standard WGAN loss
    D_gen = tf.cast ( self._discriminator (input_gen), dtype = input_gen.dtype )
    D_ref = tf.cast ( self._discriminator (input_ref), dtype = input_ref.dtype )
    d_loss = tf.reduce_mean ( w_gen * D_gen - w_ref * D_ref , axis = None )

    ## initial virtual adversarial direction
    r_k = tf.random.uniform ( shape  = tf.shape(input_ref) , 
                              minval = 0.0 , 
                              maxval = 1.0 , 
                              dtype  = input_ref.dtype )
    r_k /= tf.norm ( r_k , axis = None )

    for _ in range(self._v_adv_dir_updt):
      ## adversarial perturbation of input tensors
      input_gen_pert = tf.clip_by_value ( input_gen + self._xi * r_k , 
                                          clip_value_min = tf.reduce_min (input_gen) ,
                                          clip_value_max = tf.reduce_max (input_gen) )
      input_ref_pert = tf.clip_by_value ( input_ref + self._xi * r_k , 
                                          clip_value_min = tf.reduce_min (input_ref) ,
                                          clip_value_max = tf.reduce_max (input_ref) )

      ## approximation of virtual adversarial direction
      D_gen_pert = tf.cast ( self._discriminator (input_gen_pert), dtype = input_gen.dtype )
      D_ref_pert = tf.cast ( self._discriminator (input_ref_pert), dtype = input_ref.dtype )
      diff = tf.abs ( tf.concat ( [D_gen, D_ref], axis = 0 ) - \
                      tf.concat ( [D_gen_pert, D_ref_pert], axis = 0 ) )
      r_k = tf.gradients ( tf.reduce_mean (diff, axis = None) , r_k )[0]
      r_k /= tf.norm ( r_k , axis = None )

    ## virtual adversarial direction
    epsilon = self._epsilon_sampler ( shape = tf.shape(input_ref), dtype = input_ref.dtype )
    r_adv = epsilon * r_k

    ## adversarial perturbation of input tensors
    input_gen_pert = tf.clip_by_value ( input_gen + r_adv , 
                                        clip_value_min = tf.reduce_min (input_gen) ,
                                        clip_value_max = tf.reduce_max (input_gen) )
    input_ref_pert = tf.clip_by_value ( input_ref + r_adv , 
                                        clip_value_min = tf.reduce_min (input_ref) ,
                                        clip_value_max = tf.reduce_max (input_ref) )

    ## adversarial Lipschitz penalty correction
    D_gen_pert = tf.cast ( self._discriminator (input_gen_pert), dtype = input_gen.dtype )
    D_ref_pert = tf.cast ( self._discriminator (input_ref_pert), dtype = input_ref.dtype )
    diff = tf.abs ( tf.concat ( [D_gen, D_ref], axis = 0 ) - \
                    tf.concat ( [D_gen_pert, D_ref_pert], axis = 0 ) )
    alp_term = tf.math.maximum ( diff / tf.norm ( r_adv, axis = None ) - self._lp_const, 0.0 )   # one-side penalty
    alp_term = self._adv_lp_penalty * tf.reduce_mean (alp_term, axis = None)   # adversarial Lipschitz penalty
    d_loss += alp_term ** 2
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
    """The discriminator of the WGAN-ALP system."""
    return self._discriminator

  @property
  def generator (self) -> tf.keras.Sequential:
    """The generator of the WGAN-ALP system."""
    return self._generator

  @property
  def v_adv_dir_updt (self) -> int:
    """Number of recursive updates of the virtual adversarial direction."""
    return self._v_adv_dir_updt

  @property
  def adv_lp_penalty (self) -> float:
    """Adversarial Lipschitz penalty coefficient."""
    return self._adv_lp_penalty

  @property
  def lp_const (self) -> float:
    """Lipschitz constant of the discriminator."""
    return self._lp_const

  @lp_const.setter
  def lp_const (self, K) -> None:
    ## data-type control
    if not isinstance (K, float):
      if isinstance (K, int): K = float (K)
      else: raise TypeError ("The Lipschitz constant should be a float.")

    ## data-value control
    if K <= 0: 
      raise ValueError ("The Lipschitz constant should be greater than 0.")

    self._lp_const = K

  @property
  def epsilon_sampler (self) -> str:
    """The definition of the sampler function for the epsilon hyperparameter."""
    return inspect.getsource (self._epsilon_sampler)

  @epsilon_sampler.setter
  def epsilon_sampler (self, func) -> None:
    ## data-type control
    if not ( callable(func) and func.__name__ == "<lambda>" ):
      raise TypeError ("The epsilon sampler should be passed as a lambda function.")

    ## data-value control
    func_args = func.__code__.co_varnames
    if (len(func_args) != 2) or (func_args[0] != "shape") or (func_args[1] != "dtype"): 
      raise ValueError ( f"The lambda function for the epsilon sampler "
                         f"should have only ('shape', 'dtype') as arguments." )

    self._epsilon_sampler = func

  @property
  def xi (self) -> float:
    """The xi hyperparameter used to approximate the virtual adversarial direction."""
    return self._xi

  @xi.setter
  def xi (self, xi) -> None:
    ## data-type control
    if not isinstance (xi, float):
      if isinstance (xi, int): xi = float (xi)
      else: raise TypeError ("The xi hyperparameter should be a float.")

    ## data-value control
    if xi <= 0: 
      raise ValueError ("The xi hyperparameter should be greater than 0.")

    self._xi = xi
    