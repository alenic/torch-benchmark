import time


class Timer:
    def __init__(self):
        self._tic = None
        self._elapsed = 0.0

    def tic(self):
        self._tic = time.perf_counter()

    def toc(self):
        if self._tic is None:
            raise RuntimeError("Timer was not started with tic()")
        self._elapsed = time.perf_counter() - self._tic
        return self._elapsed

    @property
    def elapsed(self):
        if self._tic is None:
            return self._elapsed
        return time.perf_counter() - self._tic

    def __enter__(self):
        self.tic()
        return self

    def __exit__(self, *args):
        self.toc()
