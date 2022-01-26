import tensorflow as tf
from tensorflow.keras.callbacks import Callback

K = tf.keras.backend

class GanBaseLrScheduler (Callback):
  def __init__ (self):
    super().__init__()

  def on_epoch_begin (self, epoch, logs = None):
    ## Discriminator lr-scheduling
    d_lr0 = K.get_value ( self.model.d_lr0 )
    K.set_value ( self.model.d_optimizer.learning_rate, self._scheduled_lr (d_lr0, epoch) )

    ## Generator lr-scheduling
    g_lr0 = K.get_value ( self.model.g_lr0 )
    K.set_value ( self.model.g_optimizer.learning_rate, self._scheduled_lr (g_lr0, epoch) )

  def _scheduled_lr (self, lr0, epoch):
    return lr0

  def on_epoch_end (self, epoch, logs = None):
    logs = logs or {}
    logs["d_lr"] = K.get_value ( self.model.d_optimizer.learning_rate )
    logs["g_lr"] = K.get_value ( self.model.g_optimizer.learning_rate )
    