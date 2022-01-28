import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback


class ImageSaver (Callback):
  def __init__ (self, name, dirname = "./images", ext = "png", step = 1):
    super().__init__()
    self._name    = name
    self._dirname = dirname
    self._ext     = ext
    self._step    = step

    if not os.path.exists (self._dirname):
      os.makedirs (self._dirname)

  def on_epoch_end (self, epoch, logs = None):
    if (epoch + 1) % self._step == 0:
      gen_img = self.model.generate ( batch_size = 16 )
      plt.figure (figsize = (4,4), dpi = 100)
      for i in range(16):
        plt.subplot (4, 4, i+1)
        plt.imshow (gen_img[i,:,:,0], cmap = "gray")
        plt.axis ("off")
      plt.savefig (f"{self._dirname}/{self._name}_ep{epoch+1:04d}.{self._ext}")
