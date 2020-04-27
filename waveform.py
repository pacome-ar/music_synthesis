import numpy as np


### I dont know what i will need but everything is here.
# maybe creat function locally for efficiency
pi = np.pi
def sin(x):
    return np.sin(x)

def saw(x):
    return (((x - np.pi)%(2 * np.pi)) / (np.pi) - 1)

def saw_desc(x):
    return (((-x - np.pi)%(2 * np.pi)) / (np.pi) - 1)

def square(x):
    return (((x - np.pi)%(2 * np.pi) // np.pi - 0.5) * 2)

def tri(x):
    return (2 / (np.pi) * np.abs((x - np.pi/2)%(2*np.pi) - np.pi) - 1)

class Waveform():
    '''might be too slow'''
    pi = np.pi
    def __init__(self, ampl=1, sym=True,
                 phase=0, sr=1, freq=1/(2*pi)):
        self.ampl = ampl
        self.sym = sym
        self.phase = phase
        self.sr = sr
        self.freq = freq
        if not self.sym:
            self.ampl = ampl * 0.5

    def wrapp_func_creator(self, func, freq=0):
        def wrapper(x):
            return self.wrapp_func(func, x, freq)
        return wrapper

    def wrapp_func(self, func, x, freq=0):
        freq = freq + (1 - bool(freq)) * self.freq
        x = x * (2 * self.pi) * freq / self.sr + self.phase
        return self.ampl * (func(x) + (1 - self.sym))


# TODO
def make_strange_square_error():
    return Waveform(sr=44000, sym=False).wrapp_func(square, np.arange(44000), 1)


def variable_width_wf(x, freq, width, func=np.sin):
    if width == 0:
        return 0
    freq = 2 * 2*np.pi*freq
    xmod = 0.5 * (saw(x*freq - np.pi) + 1)
    funcmod = func(xmod * 2*np.pi / (2*width))
    pulsemod = pulse(xmod, width=width)
    sq = square(x*freq/2)
    return funcmod * pulsemod * sq


def pulse(x, width=1):
    if x < width:
        return 1
    else:
        return 0


def make_pulse_from(func, width=1):
    def wrapper(x):
        if x < width:
            return func(x)
        else:
            return 0
    return wrapper


def make_periodic(function, sym=False, neg=False):
    '''creates 2pi periodic function
    form a function f defined on 0, 1 (composition with lfo)'''
    sawf = Waveform(sym=False, phase=np.pi).wrapp_func_creator(saw)
    sqf = square
    trif = Waveform(sym=False, phase=-np.pi/2).wrapp_func_creator(tri)
    # tri = Waveform(ampl=0.5, center=0.5, phase=-np.pi/2).tri
    if sym:
        def wrapper(x):
            if neg:
                return sqf(x) * function(trif(x))
            else:
                return function(trif(x))
    else:
        def wrapper(x):
            return function(sawf(x))
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
