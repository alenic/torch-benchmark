import time


class Timer:
    def __init__(self):
        self._tic = time.perf_counter()
        self._elapsed = 0
    
    def tic(self):
        self._tic = time.perf_counter()

    def toc(self):
        self._elapsed = time.perf_counter()-self.tic()

    def elapsed(self):
        return self._elapsed