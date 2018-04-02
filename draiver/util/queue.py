from queue import Queue


class SkipQueue(Queue):

    def __init__(self, skip_capacity):
        super().__init__()
        self.skip_capacity = skip_capacity

    def put(self, item, block=True, timeout=None):
        with self.mutex:
            if self._qsize() >= self.skip_capacity:
                self._get()
            super()._put(item)
            self.not_empty.notify()
