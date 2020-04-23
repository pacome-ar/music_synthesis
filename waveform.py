import numpy as np

class Waveform():
    def __init__(self, ampl=1, center=0, phase=0):
        self.ampl = ampl
        self.center = center

        self.sin = lambda x : (
                        ampl * np.sin(x + phase) + center
                        )
        self.saw = lambda x : (
                        ampl * (((x + phase - np.pi)%(2 * np.pi)) / (np.pi) - 1)
                        + center
                        )
        self.square = lambda x : (
                        ampl * (
                            ((x + phase - np.pi)%(2 * np.pi) // np.pi - 0.5)
                            * 2
                            )
                        + center
                        )
        self.tri = lambda x: (
                        ampl * (
                            2 / (np.pi)
                            * np.abs((x + phase - np.pi/2)%(2*np.pi) - np.pi)
                            - 1
                            )
                    + center)

def make_periodic(function, sym=False, neg=False):
    '''creates 2pi periodic function
    form a function f defined on 0, 1 (composition with lfo)'''
    saw = Waveform(ampl=0.5, center=0.5, phase=np.pi).saw
    sq = Waveform().square
    tri = Waveform(ampl=0.5, center=0.5, phase=-np.pi/2).tri
    if sym:
        def wrapper(x):
            if neg:
                return sq(x) * function(tri(x))
            else:
                return function(tri(x))
    else:
        def wrapper(x):
            return function(saw(x))
    return wrapper

def sin_to_rect(n=1, amp=1):
    def wrapper(x):
        if n==0:
            return x * 0
        return ((4 * amp) / np.pi *
                np.array([np.sin((2*p + 1) * x) / (2*p + 1)
                          for p in range(n)]).sum(axis=0))
    return wrapper

def sin_to_tri(n=1, amp=1):
    def wrapper(x):
        if n==0:
            return x * 0
        return ((-8 * amp) / np.pi**2 *
                np.array([np.cos((2*p + 1) * x) / (2*p + 1)**2
                          for p in range(n)]).sum(axis=0))
    return wrapper
