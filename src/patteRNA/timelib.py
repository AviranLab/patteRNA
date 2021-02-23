import time
import humanfriendly


class Clock:
    def __init__(self):
        self.t = 0

    def tick(self):
        self.t = time.time()

    def tock(self, pretty=False):
        if pretty:
            return humanfriendly.format_timespan(time.time() - self.t, detailed=True)
        else:
            return time.time() - self.t
