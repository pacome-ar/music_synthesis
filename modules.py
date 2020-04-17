import numpy as np
from scipy.interpolate import interp1d
from functools import reduce
from math import copysign
import operator

def build_cst_function(val=1):
    def func(*x, **kwargs):
        return val
    return func

class NumpyQueue():
    '''Implements a queue class:
    Warning: this is a queue, the first element in on the right'''
    def __init__(self, init=None, maxsize=0, default=0):
        if init is None:
            self.size = maxsize
            self.queue = np.zeros(maxsize) * default
        elif (maxsize is 0) and (init is not None):
            self.size = len(init)
            self.queue = np.asarray(init)[::-1]
        elif (maxsize is not 0) and (init is not None):
            self.size = maxsize
            init = init[:min(len(init), maxsize)]
            self.queue = np.zeros(maxsize) * default
            self.queue[:len(init)] = init
            self.queue = self.queue[::-1]

    def prepend(self, val):
        tmp = self.queue[-1]
        self.queue[-1] = val
        self.queue = np.roll(self.queue, 1)
        return tmp

#############################

class ModuleBuilder():
    def __init__(self, name='', n_in=1, n_out=1, parameters=[],
                       function=build_cst_function(1)):
        self.name = name
        self.clock = None
        self.sr = None
        self.function = function
        ins = ['input_' + str(i+1) for i in range(n_in)]
        self.__dict__.update(zip(ins, [None] * len(ins)))
        outs = ['output_' + str(i+1) for i in range(n_out)]
        self.__dict__.update(zip(outs, [None] * len(outs)))
        params = ['param_' + p for p in parameters]
        self.__dict__.update(zip(params, [None] * len(params)))

    def __repr__(self):
        return 'module ' + self.name + ' : ' + repr(self.__dict__)

    def get_port(self, kind='input', retdict=True):
        if not retdict:
            return [k for k in self.__dict__.keys() if kind in k]
        else:
            return {k:v for k,v in self.__dict__.items() if kind in k}

    def get(self, *ports):
        return [self.__dict__[port] for port in ports]

    def __call__(self):
        inputs = self.get_port('input', retdict=True).values()
        params = self.get_port('param', retdict=True)
        val = self.function(*inputs, **params)
        for output in self.get_port('output'):
            self.__dict__[output] = val

###################

class Inputer(ModuleBuilder):
    '''translates (freq, amp) into signal'''
    def __init__(self, name='inputer', wf=np.sin):
        self.wf = wf
        super().__init__(
            name=name, n_in=1, n_out=1, function=self.translate_note)

    def translate_note(self, input_):
        freq, amp = input_
        return amp * self.wf(self.clock / self.sr * 2 * np.pi * freq)

###################

class SimpleLFO(ModuleBuilder):
    def __init__(self, name='lfo', wf=np.sin, freq=5):
        self.wf = wf
        self.freq = freq
        super().__init__(
            name=name, n_in=0, n_out=1, function=self.function
        )

    def function(self):
        return self.wf(self.clock / self.sr * 2 * np.pi * self.freq)

####################

class SimpleSaturation(ModuleBuilder):
    def __init__(self, name='simpl_sat', maxamp=1):
        self.maxamp = maxamp
        super().__init__(
            name=name, n_in=1, n_out=1, function=self.function
        )

    def function(self, input_):
        if abs(input_) < self.maxamp:
            return input_
        else:
            return copysign(1, input_) * self.maxamp

###################

class Combiner(ModuleBuilder):
    def __init__(self, name='multi', n_in=2,
                 redfunc=operator.mul, aggfunc=None):
        self.aggfunc = aggfunc
        self.redfunc = redfunc
        super().__init__(
            name=name, n_in=n_in, n_out=1, function=self.function
        )

    def function(self, *xs):
        if self.aggfunc is not None:
            return self.aggfunc(xs)
        return reduce(self.redfunc, xs)

###################

class Fir(ModuleBuilder):
    def __init__(self, name='fir', blist=[0.5, 0.5]):
        self.blist = np.asarray(blist)
        self.memx = NumpyQueue(maxsize=len(blist))
        super().__init__(
            name=name, n_in=1, n_out=1, function=self.function
        )

    def function(self, x):
        self.memx.prepend(x)
        return self.memx.queue[::-1] @ self.blist

class Iir(ModuleBuilder):
    def __init__(self, name='fir', blist=[0.5], alist=[0.5]):
        self.blist = np.asarray(blist)
        self.alist = np.asarray(alist)
        self.memx = NumpyQueue(maxsize=len(blist))
        self.memy = NumpyQueue(maxsize=len(alist))
        super().__init__(
            name=name, n_in=1, n_out=1, function=self.function
        )

    def function(self, x):
        self.memx.prepend(x)
        y = (self.memx.queue[::-1] @ self.alist
               + self.memy.queue[::-1] @ self.blist)
        self.memy.prepend(y)
        return y

###################

class SimpleDelay(ModuleBuilder):
    def __init__(self, name='delay', amp=0.5, delay=0.1):
        self.amp = amp
        self.delay = delay
        super().__init__(
            name=name, n_in=1, n_out=1, function=self.function
        )
        self._init_flag = False

    def function(self, x):
        if self._init_flag == False:
            self._init_queue()
        val = self.memo.prepend(x)
        return x + self.amp * val

    def _init_queue(self):
        assert self.sr is not None, 'Must upload the sr first'
        length = int(self.sr * self.delay)
        self.memo = NumpyQueue(maxsize=length)
        self._init_flag = True

###################

def quantbottom(x, steps):
    # bottom
    if x > steps[-1]:
        return steps[-1]
    else:
        return steps[np.argmax(x < steps)]

def quanttop(x, steps):
    # top
    if x < steps[0]:
        return steps[0]
    else:
        return steps[np.argmax(x < steps) - 1]

def quantclosest(x, steps):
    # middle(x, steps)
    return steps[np.argmin(abs(x - steps))]

class Quantizer(ModuleBuilder):
    def __init__(self, name='quant',
                 step=1, start=0, end=25, amp=1,
                 kind='closest'):
        self.step = step
        self.start = start
        self.end = end
        self.amp = amp
        self.kind = kind
        self.make_function()
        self.steps = np.arange(start, end+step, step)
        assert len(self.steps), 'no quantization steps found'
        super().__init__(
            name=name, n_in=1, n_out=1, function=self.function
        )

    def make_function(self):
        if self.kind is 'top':
            func = quanttop
        elif self.kind is 'bottom':
            func = quantbottom
        elif self.kind is 'closest':
            func = quantclosest
        else:
            raise ValueError('quantization kind not understood')
        self.function = lambda x: func(x, self.steps)

###################

class FunctionEnvelope(ModuleBuilder):
    def __init__(self, name='lfo',
                 func=build_cst_function(1)):
        self.func = func
        super().__init__(
            name=name, n_in=1, n_out=1, function=self.function
        )

    def function(self, x):
        return x * self.func(self.clock / self.sr)

class ExponentialEnvelope(FunctionEnvelope):
    def __init__(self, name='expo', decay=3):
        assert decay > 0, 'negative decay not supported'
        func = lambda x: np.exp(-decay * x)
        super().__init__(name=name, func=func)

###################

class SimpleVibrato(modules.ModuleBuilder):
    '''multiply lfo to signal'''
    def __init__(self, name='vib', wf=np.sin, freq=5):
        self.wf = wf
        self.freq = freq
        super().__init__(
            name=name, n_in=1, n_out=1, function=self.function
        )

    def function(self, x):
        return x * self.wf(
            self.clock / self.sr * 2 * np.pi * self.freq)
