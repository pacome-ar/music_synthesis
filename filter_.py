import numpy as np
import modules
import sympy
from sympy.parsing.sympy_parser import parse_expr
from itertools import permutations
from math import factorial

###########################
# FILTER BUILDER HELPERS

def Hs_to_xylists(Hfunc, astype=float):
    '''function that gets the IIR coefs of the digital filter
    from the transfert function of an analog filter Hfunc
    The analog transfer function is
        H(z) = (xlist @ zlist) / (1 - ylist @ zlist[1:])
    with zlist = [z^0, z^-1, z^-2, ...]
    works only if resulting transfert function as same degree
    on numerator and denominator
    '''
    z = sympy.symbols('z')
    s = 2*(z - 1) / (z + 1)
    H = sympy.simplify(Hfunc(s))
    num, denom = (parse_expr(x) for x in str(H).split('/'))

    # get coefficients in of the polynoms in z
    # they are given with highest order first
    numcoefs = np.array(sympy.poly(num, z).all_coeffs())
    denomcoefs = np.array(sympy.poly(denom, z).all_coeffs())
    xlist, ylist = z_to_1oz(numcoefs, denomcoefs)
    return np.asarray(xlist, dtype=astype), np.asarray(ylist, dtype=astype)


def z_to_1oz(numcoefs, denomcoefs):
    '''numcoefs and denom coefs given highest order first
    num = numcoefs[-1] + numcoefs[-2] z + ...
    '''
    N, M = len(numcoefs), len(denomcoefs)
    if N > M and M != 1:
        print('Cant find IIR params with this inputs')
        raise ValueError

    norm = denomcoefs[0]
    if M == 1:
        xlist = numcoefs / norm
        ylist = []
    elif M == N:
        xlist = numcoefs / norm
        ylist = -denomcoefs[1:] / norm
    elif M > N:
        norm = denomcoefs[0]
        xlist = [0] * (M - N) + list(numcoefs / norm)
        ylist = -denomcoefs[1:] / norm
    return xlist, ylist


def newton_to_mono(zeros):
    '''returns the coeffiscients of a polynomial in the monomial base
    from a list of zeros:
        f(x) = \prod_{i=0}^N (x - z_i)
             = \sum_{i=0}^{N+1} a_i x_i
    '''
    ret = [
        (-1)**i
        * np.sum(
            [np.prod(e) for e in permutations(zeros, i)]
        ) / factorial(i)
        for i in range(len(zeros) + 1)
    ]
    return ret


def func_from_zpg(zeros=[], poles=[], gain=1):
    N, M = len(zeros), len(poles)
    def wrapper(x):
        num = np.prod([(x - z) for z in zeros])
        denom = np.prod([(x - p) for p in poles])
        if N == 0:
            num = 1
        if M == 0:
            denom = 1
        return gain * num / denom
    return wrapper


def func_from_poly(numcoefs=[1], denomcoefs=[1]):
    '''coeffs gven highest order first'''
    N, M = len(numcoefs), len(denomcoefs)
    def wrapper(x):
        num = np.sum(
            [(b * x**(N-i)) for i, b in enumerate(numcoefs)])
        denom = np.sum(
            [(a * x**(M-i)) for i, a in enumerate(denomcoefs)])
        return num / denom
    return wrapper

#####################################################
# Filter functions

def make_low_pass_coeffs(w):
    a = w / (1 + w)
    xlist, ylist = [a], [1 - a]
    return xlist, ylist

def make_high_pass_coeffs(w):
    a = 1 / (1 + w)
    xlist, ylist = [a, -a], [a]
    return xlist, ylist

def make_band_pass_coeffs(w, Q=0.1):
    alpha = (1 + 2*Q/w + Q*w/2)
    beta = (Q*w - 4*Q/w)
    gamma = (2*Q/w + Q*w/2 - 1)
    xlist = [1/alpha, 0, -1/alpha]
    ylist = [-beta/alpha, -gamma/alpha]
    return xlist, ylist

def make_band_stop_coeffs(w0, wc=0.1):
    xlist = np.array([w0**2 + 4, 2*w0**2 - 8, w0**2 + 4])
    ylist = np.array([2*w0**2 - 8, w0**2 - 2*wc + 4])
    norm = (w0**2 + 2*wc + 4)
    xlist = xlist / norm
    ylist = -ylist / norm
    return xlist, ylist

######################################################

class Iir():
    def __init__(self, xlist=[], ylist=[]):
        self.xlist = np.asarray(xlist)
        self.ylist = np.asarray(ylist)
        self.memx = [0] * len(xlist)
        self.memy = [0] * len(ylist)

    def __call__(self, x):
        self.memx = [x] + self.memx[:-1]
        y = np.dot(self.memx, self.xlist)
        y += np.dot(self.memy, self.ylist)
        self.memy = [y] + self.memy[:-1]
        return y

    def update_xylist(self, xlist=None, ylist=None):
        '''allows to change the value of xlist
        without while keeping the memory
        Warning, new xlist must the same size as old'''
        if xlist:
            self.xlist = xlist
        if ylist:
            self.ylist = ylist
        self.xlist = xlist

###################################################

class FilterA(modules.ModuleBuilder):
    '''simple 1 pole low pass filter'''
    def __init__(self, name='filterA1'):
        super().__init__(
            name=name,
            ins=['Input'], params=['frequency'],
            outs=['Output'],
            function=self.filter_)
        self.filter_freq = 504
        self.iir_flag = False

    def make_filter(self, frequency):
        if frequency == self.filter_freq:
            return
        else:
            self.filter_freq = frequency
            w = 2*np.pi * frequency / self.sr
            a = w / (1 + w)
            if not self.iir_flag:
                self.iir = Iir(ylist=[(1 - a)], xlist=[a])
                self.iir_flag = True
            else:
                self.iir.update_xylist(ylist=[(1 - a)], xlist=[a])

    def filter_(self, Input=0, frequency=504):
        Input, frequency = self._get_default_value(
            [Input, frequency], [0, 504]
        )
        self.make_filter(frequency)
        Output = self.iir(Input)
        return {'Output':Output}

###################################################

class FilterB(modules.ModuleBuilder):
    '''simple 1 pole high pass filter'''
    def __init__(self, name='filterB1'):
        super().__init__(
            name=name,
            ins=['Input'], params=['frequency'],
            outs=['Output'],
            function=self.filter_)
        self.filter_freq = 504
        self.iir_flag = False

    def make_filter(self, frequency):
        if frequency == self.filter_freq:
            return
        else:
            self.filter_freq = frequency
            w = 2*np.pi * frequency / self.sr
            a = 1 / (1 + w)
            if not self.iir_flag:
                self.iir = Iir(ylist=[a], xlist=[a, -a])
                self.iir_flag = True
            else:
                self.iir.update_xylist(ylist=[a], xlist=[a, -a])

    def filter_(self, Input=0, frequency=504):
        Input, frequency = self._get_default_value(
            [Input, frequency], [0, 504]
        )
        self.make_filter(frequency)
        Output = self.iir(Input)
        return {'Output':Output}

###################################################

class BandPass(modules.ModuleBuilder):
    '''simple 1 pole high pass filter'''
    def __init__(self, name='bandpass1'):
        super().__init__(
            name=name,
            ins=['Input'], params=['frequency', 'Q'],
            outs=['Output'],
            function=self.filter_)
        self.filter_freq = 504
        self.filter_res = 0.1
        self.iir_flag = False

    def make_filter(self, frequency, Q):
        if (frequency == self.filter_freq
            and Q == self.filter_res):
            return
        else:
            self.filter_freq = frequency
            self.filter_res = Q
            w = 2*np.pi * frequency / self.sr
            alpha = (1 + 2*Q/w + Q*w/2)
            beta = (Q*w - 4*Q/w)
            gamma = (2*Q/w + Q*w/2 - 1)
            xlist = [1/alpha, 0, -1/alpha]
            ylist = [-beta/alpha, -gamma/alpha]
            if not self.iir_flag:
                self.iir = Iir(ylist=ylist, xlist=xlist)
                self.iir_flag = True
            else:
                self.iir.update_xylist(ylist=ylist, xlist=xlist)

    def filter_(self, Input=0, frequency=504, Q=0.1):
        Input, frequency, Q = self._get_default_value(
            [Input, frequency, Q], [0, 504, 0.1]
        )
        self.make_filter(frequency, Q)
        Output = self.iir(Input)
        return {'Output':Output}

##################################################

class BandStop(modules.ModuleBuilder):
    '''simple 1 pole high pass filter'''
    def __init__(self, name='bandpass1'):
        super().__init__(
            name=name,
            ins=['Input'], params=['frequency', 'wc'],
            outs=['Output'],
            function=self.filter_)
        self.filter_freq = 504
        self.filter_res = 0.1
        self.iir_flag = False

    def make_filter(self, frequency, wc):
        if (frequency == self.filter_freq
            and wc == self.filter_res):
            return
        else:
            self.filter_freq = frequency
            self.filter_res = wc
            w0 = 2*np.pi * frequency / self.sr
            xlist = np.array([w0**2 + 4, 2*w0**2 - 8, w0**2 + 4])
            ylist = np.array([2*w0**2 - 8, w0**2 - 2*wc + 4])
            norm = (w0**2 + 2*wc + 4)
            xlist = xlist / norm
            ylist = -ylist / norm
            if not self.iir_flag:
                self.iir = Iir(ylist=ylist, xlist=xlist)
                self.iir_flag = True
            else:
                self.iir.update_xylist(ylist=ylist, xlist=xlist)

    def filter_(self, Input=0, frequency=504, wc=0.1):
        Input, frequency, wc = self._get_default_value(
            [Input, frequency, wc], [0, 504, 0.1]
        )
        self.make_filter(frequency, wc)
        Output = self.iir(Input)
        return {'Output':Output}


##################################################
################  TESTING  #######################
##################################################
import random

def test_filter(freq=500, sr=41000, filterobj=FilterA()):

    xx = [random.gauss(0, 1) for i in range(2*sr)]
    filterobj.frequency = freq
    filterobj.sr = sr

    xfilt = []
    for i, x in enumerate(xx):
        filterobj.Input = x
        filterobj()
        xfilt.append(filterobj.Output)

    fft = np.fft.fft(xx)[:len(xx)//2]
    fftfilt = np.fft.fft(xfilt)[:len(xx)//2]
    fftfreqs = np.fft.fftfreq(len(xfilt), d=1/sr)[:len(xx)//2]

    return fft, fftfilt, fftfreqs
