from collections import deque
import numpy as np

class FrameStack:

    def __init__(self, stack_size=4):
        self.frames = deque(maxlen=stack_size)

    def reset(self, frame):
        self.frames.clear()
        for _ in range(4):
            self.frames.append(frame)
        return self.get()

    def append(self, frame):
        self.frames.append(frame)
        return self.get()

    def get(self):
        return np.concatenate(list(self.frames), axis=0)