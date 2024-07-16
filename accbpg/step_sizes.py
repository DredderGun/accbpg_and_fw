class StepSize:
    def __call__(self, x, h, k, *args, **kwargs):
        pass


class ConstantStepSize(StepSize):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, x, h, k, *args, **kwargs):
        return self.alpha