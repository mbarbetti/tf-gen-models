import tensorflow as tf


d_loss_tracker = tf.keras.metrics.Mean ( name = "d_loss" )
"""Metric instance to track the discriminator loss score."""

g_loss_tracker = tf.keras.metrics.Mean ( name = "g_loss" )
"""Metric instance to track the generator loss score."""

mse_tracker = tf.keras.metrics.MeanSquaredError ( name = "mse" )
"""Metric instance to track the mean square error."""


class GAN (tf.keras.Model):
  def __init__ ( self , 
                 generator     ,
                 discriminator , 
                 input_shape  = None ,
                 output_shape = None ,
                 latent_dim = 128 ) -> None:
    super().__init__()
    self._loss_name = "Loss function"

    ## data-type control
    if input_shape is not None:
      if not isinstance (input_shape, tuple):
        raise TypeError ("The input shape should be passed as tuple.")

    if output_shape is not None:
      if not isinstance (output_shape, tuple):
        raise TypeError ("The output shape should be passed as tuple.")

    if not isinstance (latent_dim, int):
      raise TypeError ("The latent space dimension should be an integer.")

    ## data-value control
    if output_shape is None:
      output_shape = generator.output_shape[1:]
    else:
      if output_shape != tuple ( generator.output_shape[1:] ):
        raise ValueError ( f"The output shape passed doesn't match with " 
                           f"the output shape of the generator." )

    if latent_dim <= 0:
      raise ValueError ("The latent space dimension should be greater than 0.")

    self._generator     = generator
    self._discriminator = discriminator
    self._input_shape   = input_shape
    self._output_shape  = output_shape
    self._latent_dim    = latent_dim

  def compile ( self , 
                g_optimizer ,
                d_optimizer , 
                g_updt_per_batch = 1 ,
                d_updt_per_batch = 1 ) -> None:
    super().compile()

    ## compute input shapes
    if self._input_shape is None:
      g_input_shape = ( None, self._latent_dim )
      d_input_shape = tuple ( [None] + list(self._output_shape) )
    else:
      raise NotImplementedError ("This feature is not yet implemented!")   # TODO implement conditional GAN
    
    ## build training players
    self._generator     . build ( input_shape = g_input_shape )
    self._discriminator . build ( input_shape = d_input_shape )

    ## data-type control
    if not isinstance (g_updt_per_batch, int):
      if isinstance (g_updt_per_batch, float): int (g_updt_per_batch)
      else: raise TypeError ("The number of generator updates per batch should be an integer.")

    if not isinstance (d_updt_per_batch, int):
      if isinstance (d_updt_per_batch, float): int (d_updt_per_batch)
      else: raise TypeError ("The number of discriminator updates per batch should be an integer.")

    ## data-value control
    if d_updt_per_batch <= 0:
      raise ValueError ("The number of discriminator updates per batch should be greater than 0.")
    if g_updt_per_batch <= 0:
      raise ValueError ("The number of generator updates per batch should be greater than 0.")

    self._g_optimizer = g_optimizer
    self._d_optimizer = d_optimizer
    self._g_lr0 = float ( g_optimizer.learning_rate )
    self._d_lr0 = float ( d_optimizer.learning_rate )
    self._g_updt_per_batch = g_updt_per_batch
    self._d_updt_per_batch = d_updt_per_batch

  def summary (self) -> None:
    """Print a string summary of the generator and discriminator networks."""
    self._generator     . summary()
    self._discriminator . summary()

  def train_step (self, data) -> dict:
    """Train step for Keras APIs."""
    X, Y, w = self._unpack_data (data)

    ## discriminator updates per batch
    for i in range(self._d_updt_per_batch):
      self._train_d_step (X, Y, w)

    ## generator updates per batch
    for j in range(self._g_updt_per_batch):
      self._train_g_step (X, Y, w)

    ## loss computation
    ref_sample, gen_sample = self._arrange_samples (X, Y, w)
    d_loss = self._compute_d_loss (gen_sample, ref_sample)
    g_loss = self._compute_g_loss (gen_sample, ref_sample)
    threshold = self._compute_threshold (ref_sample)

    ## update metrics state
    d_loss_tracker . update_state (d_loss + threshold)
    g_loss_tracker . update_state (g_loss - threshold)

    Y_gen = self.generate ( X, batch_size = tf.shape(Y)[0] )
    mse_tracker . update_state (Y, Y_gen, sample_weight = w)

    return { "mse"    : mse_tracker.result()    ,
             "d_loss" : d_loss_tracker.result() ,
             "g_loss" : g_loss_tracker.result() ,
             "d_lr"   : self._d_optimizer.lr    ,
             "g_lr"   : self._g_optimizer.lr    } 

  def test_step (self, data) -> dict:
    """Test step for Keras APIs."""
    X, Y, w = self._unpack_data (data)

    ## loss computation
    ref_sample, gen_sample = self._arrange_samples (X, Y, w)
    d_loss = self._compute_d_loss (gen_sample, ref_sample)
    g_loss = self._compute_g_loss (gen_sample, ref_sample)
    threshold = self._compute_threshold (ref_sample)

    ## update metrics state
    d_loss_tracker . update_state (d_loss + threshold)
    g_loss_tracker . update_state (g_loss - threshold)

    Y_gen = self.generate ( X, batch_size = tf.shape(Y)[0] )
    mse_tracker . update_state (Y, Y_gen, sample_weight = w)

    return { "mse"    : mse_tracker.result()    ,
             "d_loss" : d_loss_tracker.result() ,
             "g_loss" : g_loss_tracker.result() ,
             "d_lr"   : self._d_optimizer.lr    ,
             "g_lr"   : self._g_optimizer.lr    } 

  def _unpack_data (self, data) -> tuple:
    """Unpack data-batch into generator input-output and weights (`None`, if not available)."""
    ## STANDARD GAN
    if self._input_shape is None:
      X = None
      if isinstance (data, tuple):
        Y , w = data
      else:
        Y = data
        w = None

    ## CONDITIONAL GAN
    else:
      if len(data) == 3:
        X , Y , w = data
      else:
        X , Y = data
        w = None

    return X , Y , w

  def _arrange_samples (self, X, Y, w = None) -> tuple:
    ## STANDARD GAN
    if self._input_shape is None:
      batch_size = tf.cast ( tf.shape(Y)[0], tf.int32 )

      ## map the latent space into the generated space
      input_tensor = tf.random.normal ( shape = (batch_size, self._latent_dim), dtype = Y.dtype )
      Y_gen = tf.cast ( self._generator (input_tensor), dtype = Y.dtype )

      ## weights default
      if w is None:
        w = tf.ones ( batch_size, dtype = Y.dtype )

      ## generated and reference samples
      gen_sample = ( Y_gen , w )
      ref_sample = ( Y , w )

    ## CONDITIONAL GAN
    else:
      batch_size = tf.cast ( tf.shape(Y)[0] / 2, tf.int32 )
 
      ## data-batch splitting
      X_ref , X_gen = X[:batch_size], X[batch_size:batch_size*2]
      Y_ref = Y[:batch_size]
      if w is not None:
        w_ref, w_gen = w[:batch_size], w[batch_size:batch_size*2]
      else:
        w_ref = tf.ones ( batch_size, dtype = Y_ref.dtype )
        w_gen = tf.ones ( batch_size, dtype = Y_ref.dtype )
  
      ## map the latent space into the generated space
      latent_tensor = tf.random.normal ( shape = (batch_size, self._latent_dim), dtype = Y.dtype )
      input_tensor  = tf.concat ( [X_gen, latent_tensor], axis = 1 )
      Y_gen = tf.cast ( self._generator (input_tensor), dtype = Y.dtype )
  
      ## tensors combination
      XY_gen = tf.concat ( [X_gen, Y_gen], axis = 1 )
      XY_ref = tf.concat ( [X_ref, Y_ref], axis = 1 )
  
      ## generated and reference samples
      gen_sample = ( XY_gen, w_gen )
      ref_sample = ( XY_ref, w_ref )
      
    return gen_sample, ref_sample

  @tf.function
  def _train_d_step (self, X, Y, w = None) -> None:
    with tf.GradientTape() as tape:
      gen_sample, ref_sample = self._arrange_samples (X, Y, w)
      d_loss = self._compute_d_loss ( gen_sample, ref_sample )
    grads = tape.gradient ( d_loss, self._discriminator.trainable_weights )
    self._d_optimizer.apply_gradients ( zip (grads, self._discriminator.trainable_weights) )

  def _compute_d_loss (self, gen_sample, ref_sample) -> tf.Tensor:
    return - self._compute_g_loss (gen_sample, ref_sample)

  tf.function
  def _train_g_step (self, X, Y, w = None) -> None:
    with tf.GradientTape() as tape:
      gen_sample, ref_sample = self._arrange_samples (X, Y, w)
      g_loss = self._compute_g_loss ( gen_sample, ref_sample )
    grads = tape.gradient ( g_loss, self._generator.trainable_weights )
    self._g_optimizer.apply_gradients ( zip (grads, self._generator.trainable_weights) )

  def _compute_g_loss (self, gen_sample, ref_sample) -> tf.Tensor:
    ## extract features and weights
    input_gen, w_gen = gen_sample
    input_ref, w_ref = ref_sample

    ## noise injection to stabilize GAN training
    rnd_gen = tf.random.normal ( tf.shape (input_gen), stddev = 0.1, dtype = input_gen.dtype )
    rnd_ref = tf.random.normal ( tf.shape (input_ref), stddev = 0.1, dtype = input_ref.dtype )
    D_gen = tf.cast ( self._discriminator (input_gen + rnd_gen), dtype = input_gen.dtype )
    D_ref = tf.cast ( self._discriminator (input_ref + rnd_ref), dtype = input_ref.dtype )

    ## loss computation
    g_loss = w_ref * tf.math.log ( tf.clip_by_value ( D_ref       , 1e-12 , 1.0 ) ) + \
             w_gen * tf.math.log ( tf.clip_by_value ( 1.0 - D_gen , 1e-12 , 1.0 ) )
    return tf.reduce_mean (g_loss, axis = None)

  def _compute_threshold (self, ref_sample) -> tf.Tensor:
    ## extract features and weights
    input_ref, w_ref = ref_sample

    ## noise injection to stabilize GAN training
    rnd_ref = tf.random.normal ( tf.shape (input_ref), stddev = 0.1, dtype = input_ref.dtype )
    D_ref = tf.cast ( self._discriminator (input_ref + rnd_ref), dtype = input_ref.dtype )

    ## split features and weights
    batch_size = tf.cast ( tf.shape(input_ref)[0] / 2, tf.int32 )
    D_ref_1, D_ref_2 = D_ref[:batch_size], D_ref[batch_size:batch_size*2]
    w_ref_1, w_ref_2 = w_ref[:batch_size], w_ref[batch_size:batch_size*2]

    ## threshold loss computation
    th_loss = w_ref_1 * tf.math.log ( tf.clip_by_value ( D_ref_1       , 1e-12 , 1.0 ) ) + \
              w_ref_2 * tf.math.log ( tf.clip_by_value ( 1.0 - D_ref_2 , 1e-12 , 1.0 ) )
    return tf.reduce_mean (th_loss, axis = None)

  def generate (self, X = None, batch_size = None) -> tf.Tensor:
    ## data-value control
    if (X is None) and (batch_size is None):
      raise ValueError ( f"One among the condition tensor X or the batch-size " 
                         f"should be passed to compute the generator output." )
    elif (X is not None) and (batch_size is not None):
      if tf.shape(X)[0] != batch_size:
        raise ValueError ( f"When the batch-size is passed, it should match with "
                           f"the number of rows of the condition tensor X." )
    elif (X is not None) and (batch_size is None):
      batch_size = tf.shape(X)[0]

    ## map the latent space into the generated space
    input_tensor = tf.random.normal ( shape = (batch_size, self._latent_dim) )
    if X is not None: input_tensor = tf.concat ( [X, input_tensor], axis = 1 )   # for Conditional GAN
    Y_gen = self.generator (input_tensor)
    return Y_gen

  @property
  def loss_name (self) -> str:
    """Name of the loss function used for training."""
    return self._loss_name

  @property
  def generator (self) -> tf.keras.Sequential:
    """The generator of the GAN system."""
    return self._generator

  @property
  def discriminator (self) -> tf.keras.Sequential:
    """The discriminator of the GAN system."""
    return self._discriminator

  @property
  def latent_dim (self) -> int:
    """The dimension of the latent space."""
    return self._latent_dim

  @property
  def g_optimizer (self) -> tf.keras.optimizers.Optimizer:
    """The generator optimizer.."""
    return self._g_optimizer

  @property
  def d_optimizer (self) -> tf.keras.optimizers.Optimizer:
    """The discriminator optimizer."""
    return self._d_optimizer

  @property
  def d_lr0 (self) -> float:
    """Initial value for discriminator learning rate."""
    return self._d_lr0

  @property
  def g_lr0 (self) -> float:
    """Initial value for generator learning rate."""
    return self._g_lr0

  @property
  def g_updt_per_batch (self) -> int:
    """Number of generator updates per batch."""
    return self._g_updt_per_batch

  @property
  def d_updt_per_batch (self) -> int:
    """Number of discriminator updates per batch."""
    return self._d_updt_per_batch

  @property
  def metrics (self) -> list:
    return [d_loss_tracker, g_loss_tracker, mse_tracker]
