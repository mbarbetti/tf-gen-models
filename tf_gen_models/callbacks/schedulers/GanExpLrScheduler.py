from tf_gen_models.callbacks.schedulers import GanBaseLrScheduler

class GanExpLrScheduler (GanBaseLrScheduler):
  def __init__ (self, factor = 0.1, step = 1):
    super().__init__()
    self._factor = factor
    self._step = step

  def _scheduled_lr (self, lr0, epoch):
    return lr0 * self._factor ** (epoch / self._step)
