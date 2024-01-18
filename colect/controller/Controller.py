
class Controller:
    """Controller (Base class)

    Args:
        frequency (float): Controller frequency, defaults to 500Hz
    """

    def __init__(self, frequency=500.0):
        self._frequency = frequency
        self._dt = 1 / self._frequency

    def run_controller(self):
        raise NotImplementedError("run_controller() is not implemented in Controller base class")

    def get_output(self):
        raise NotImplementedError("get_output() is not implemented in Controller base class")

    def start(self):
        raise NotImplementedError("start() is not implemented in Controller base class")

    def stop(self):
        raise NotImplementedError("stop() is not implemented in Controller base class")

    def reset(self):
        raise NotImplementedError("reset() is not implemented in Controller base class")

    def set_default_controller_parameters(self):
        raise NotImplementedError("set_default_controller_parameters() is not implemented in Controller base class")
