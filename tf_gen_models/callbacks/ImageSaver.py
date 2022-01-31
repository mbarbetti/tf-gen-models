import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback


class ImageSaver (Callback):
  def __init__ ( self , 
                 name , 
                 dirname = "./images" , 
                 ext = "png" , 
                 step = 1 ,
                 num_images = 16 ) -> None:
    super().__init__()
    self._filename = f"{dirname}/{name}"

    if not os.path.exists (dirname):
      os.makedirs (dirname)

    self._ext = ext
    self._step = step
    self._num_images = num_images

  def on_epoch_end (self, epoch, logs = None) -> None:
    if (epoch + 1) % self._step == 0:
      gen_img = self.model.generate ( batch_size = self._num_images )

      if self._num_images % 2 == 0:
        rows = self._num_images / 2
        cols = self._num_images / 2
      else:
        rows = 1 ; cols = self._num_images

      plt.figure (figsize = (4,4), dpi = 100)
      for i in range(self._num_images):
        plt.subplot (rows, cols, i+1)
        plt.imshow (gen_img[i,:,:,0], cmap = "gray")
        plt.axis ("off")
      plt.savefig (f"{self._filename}_ep{epoch+1:04d}.{self._ext}")
      plt.close()
