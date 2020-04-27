# A place to put all helper functions and classes

import option
import time

class Timer:
    'A helper class to print passed time.'
    def __init__(self):
        self.reset()

    def __call__(self):
        delta = int(time.time() - self.start_time)
        h = delta//3600
        m = (delta % 3600) // 60
        s = delta % 60
        result = ""
        if h > 0:
            result += f"{h} hours "
        if h > 0 or m > 0:
            result += f"{m} minutes "
        result += f"{s} seconds"
        return result

    def reset(self):
        self.start_time = time.time()
