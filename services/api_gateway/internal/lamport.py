from threading import Lock


class LamportClock:
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = Lock()

    def next(self) -> int:
        with self._lock:
            self._value += 1
            return self._value
