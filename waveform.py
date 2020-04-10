import numpy as np

class Waveform():
    def __init__(self):
        self.sin = np.sin
        self.saw = lambda x : ((x - np.pi)%(2 * np.pi)) / (np.pi) - 1
        self.square = lambda x : ((x - np.pi)%(2 * np.pi) // np.pi - 0.5) * 2
        self.tri = lambda x: 2 / (np.pi) * np.abs((x - np.pi/2)%(2*np.pi) - np.pi) - 1
