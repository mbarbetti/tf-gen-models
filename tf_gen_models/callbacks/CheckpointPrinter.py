import os
from datetime import datetime
from tensorflow.keras.callbacks import Callback


class CheckpointPrinter (Callback):
  def __init__ (self, step = 1) -> None:
    super().__init__()
    self._step = step

  def on_train_begin (self, logs = None) -> None:
    self._start = datetime.now()
    print (f"{self._start} | Begin of the training procedure")

  def on_train_end (self, logs = None) -> None:
    stop = datetime.now()
    timestamp = str(stop - self._start) . split (".") [0]   # HH:MM:SS
    timestamp = timestamp . split (":")   # [HH, MM, SS]
    timestamp = f"{timestamp[0]}h {timestamp[1]}min {timestamp[2]}s"
    print (f"{stop} | End of the training procedure | Duration: {timestamp}")

  def on_epoch_end (self, epoch, logs = None) -> None:
    if (epoch + 1) % self._step == 0:
      scores = ""
      for item in logs.items():
        scores += f"{item[0]}: {item[1]:.2e} - "
      print (f"{datetime.now()} | Epoch: {epoch+1} | {scores[:-3]} |")
      