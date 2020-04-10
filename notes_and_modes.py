import numpy as np
from collections import namedtuple

#######

fullOctave = ['A', 'As', 'B', 'C', 'Cs', 'D', 'Ds', 'E', 'F', 'Fs', 'G', 'Gs']

def make_octave(start='A', num=4):
    ret = np.roll(fullOctave, -np.where(fullOctave == 'C')[0])
    return np.array([note + '_' + str(num) for note in ret])

#######

class Mode():
    def __init__(self, intervals):
        self.intervals = np.concatenate(([0], intervals))
        self.N = len(self.intervals)
        self.intervals_idx = np.cumsum(self.intervals * 2, dtype='int')

    def _degreeToIndex(self, degree):
        deg = degree - 1
        return 12 * (deg // self.N) + self.intervals_idx[deg % self.N]

    def _intervalToIndex(self, interval):
        N = self.N 
        return 12 * (interval // N) + self.intervals_idx[interval % N]

class Modes():
    def __init__(self):
        self.major = Mode([1, 1, 1/2, 1, 1, 1])
        self.minor = Mode([1, 1/2, 1, 1, 1/2, 1])
        self.ionian = self.major
        self.dorian = Mode([1, 1/2, 1, 1, 1, 1/2])
        self.phrygian = Mode([1/2, 1, 1, 1, 1/2, 1])
        self.lydian = Mode([1, 1, 1, 1/2, 1, 1])
        self.mixolydian = Mode([1, 1, 1/2, 1, 1, 1/2])
        self.aeolian = Mode([1, 1/2, 1, 1, 1/2, 1])
        self.locrian = Mode([1/2, 1, 1, 1/2, 1, 1])
        self.diatoniques = ['ionian', 'dorian', 'phrygian', 'lydian',
                            'mixolydian', 'aeolian', 'locrian']
