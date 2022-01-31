import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback


class ImageSaver (Callback):
  def __init__ ( self , 
                 name , 
                 dirname = "./images" , 
                 ext = "png" , 
                 step = 1 ,
                 look = "multi" ) -> None:
    super().__init__()
    self._filename = f"{dirname}/{name}"

    if not os.path.exists (dirname):
      os.makedirs (dirname)

    self._ext = ext
    self._step = step

    if look not in ["single", "multi"]:
      raise ValueError ("`look` should be chosen in ['single', 'multi'].")

    self._look = look

  def on_epoch_end (self, epoch, logs = None) -> None:
    if (epoch + 1) % self._step == 0:
      if self._look == "single":
        rows = 1 ; cols = 1 ; batch_size = 1
      else:
        rows = 4 ; cols = 4 ; batch_size = 16

      gen_img = self.model.generate ( batch_size = batch_size )

      plt.figure (figsize = (4,4), dpi = 100)
      for i in range(batch_size):
        plt.subplot (rows, cols, i+1)
        plt.imshow (gen_img[i,:,:,0], cmap = "gray")
        plt.axis ("off")
      plt.savefig (f"{self._filename}_ep{epoch+1:04d}.{self._ext}")
      plt.close()
