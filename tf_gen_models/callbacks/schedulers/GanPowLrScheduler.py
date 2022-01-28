from tf_gen_models.callbacks.schedulers import GanBaseLrScheduler

class GanPowLrScheduler (GanBaseLrScheduler):
  def __init__ (self, decay = 1, step = 1):
    super().__init__()
    self._decay = decay
    self._step = step

  def _scheduled_lr (self, lr0, epoch):
    return lr0 / (1.0 + self._decay * epoch / self._step)
