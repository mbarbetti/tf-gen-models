import os
from tensorflow.keras.callbacks import Callback


class ModelSaver (Callback):
  def __init__ (self, name, dirname = "./models", saved_model = "both", step = None) -> None:
    super().__init__()
    self._filename = f"{dirname}/{name}"

    if not os.path.exists (f"{self._filename}"):
      os.makedirs (f"{self._filename}")

    if saved_model not in ["gen", "disc", "both"]:
      raise ValueError ("`saved_model` should be chosen in ['gen', 'disc', 'both'].")

    self._saved_model = saved_model
    self._step = step

  def on_train_end (self, logs = None) -> None:
    if self._saved_model == "gen":
      self.model.generator . save ( f"{self._filename}/saved_gen_latest", save_format = "tf" )
    elif self._saved_model == "disc":
      self.model.discriminator . save ( f"{self._filename}/saved_disc_latest", save_format = "tf" )
    else:
      self.model.generator . save ( f"{self._filename}/saved_gen_latest", save_format = "tf" )
      self.model.discriminator . save ( f"{self._filename}/saved_disc_latest", save_format = "tf" )    

  def on_epoch_end (self, epoch, logs = None) -> None:
    if self._step is not None:
      if (epoch + 1) % self._step == 0:
        if self._saved_model == "gen":
          self.model.generator . save ( f"{self._filename}/saved_gen_ep{epoch+1:04d}", save_format = "tf" )
        elif self._saved_model == "disc":
          self.model.discriminator . save ( f"{self._filename}/saved_disc_ep{epoch+1:04d}", save_format = "tf" )
        else:
          self.model.generator . save ( f"{self._filename}/saved_gen_ep{epoch+1:04d}", save_format = "tf" )
          self.model.discriminator . save ( f"{self._filename}/saved_disc_ep{epoch+1:04d}", save_format = "tf" )
