import numpy as np
from scipy.interpolate import interp1d
from functools import reduce
from math import copysign
import operator
import random
import waveform

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

def make_default_function(ins, params, outs, val=1):
    inputs = ins + params
    vals = [None] * len(inputs)
    default_kwargs = dict(zip(inputs, vals))
    def wrapper(**default_kwargs):
        outputs = dict(zip(outs, [val]*len(outs)))
        return outputs
    return wrapper

class ModuleBuilder():
    '''Generic class to build modules

    attrs:
    ------
    name: str
        name of the module
    ins: list of str
        list of the inputs of the module
        an input is a value that is to be set during execution
    params: list of str
        list of the parameters of the module
        a parameter has an internal value, which
        can be set dynamically by a user
    outs: list of str
        list of the outputs of the module
        outputs are the result of calculations throughout the module
    function: function
        the internal function:
            takes a dictionnary of ins and params
            returns a dictionnary of outs
    '''
    def __init__(self, name='module',
                       ins=['input_1'], params=[], outs=['output_1'],
                       function=None,
                       sr=44000):
        '''init function'''
        self.name = name
        self.clock = 0
        self.sr = sr
        if function is None:
            self.function = make_default_function(ins, params, outs)
        else:
            self.function = function
        self.ins = ins
        self.params = params
        self.outs = outs
        self._make_ports(ins)
        self._make_ports(outs)
        self._make_ports(params)

    def initialize(self):
        '''does precalculations before the call
        when a module is added to a synth
        usefull if the moduels needs to know informations about the
        the synth
        '''
        pass

    def _assert_range(self, buttonval, buttonname, minval, maxval):
        assert minval <= buttonval <= maxval, \
            buttonname + f' should be between {minval} and {maxval}'

    def _get_default_value(self, params, values):
        return [
            param if param else val for param, val in zip(params, values)
    ]

    def _make_ports(self, ports):
        vals = [None] * len(ports)
        self._update_ports(dict(zip(ports, vals)))

    def _get_ports(self, *ports):
        return [self.__dict__[port] for port in ports]

    def _get_ports_by_cat(self, cat='ins', valonly=False):
        listport = self.__dict__[cat]
        if valonly:
            return self._get_port(*listport)
        return dict((p, self.__dict__[p]) for p in listport)

    def _update_ports(self, updatedict):
        self.__dict__.update(updatedict)

    def __repr__(self):
        return 'module ' + self.name + ' : ' + repr(self.__dict__)

    def __call__(self):
        ins = self._get_ports_by_cat('ins')
        params = self._get_ports_by_cat('params')
        inputs = {**ins, **params}
        outs = self.function(**inputs)
        self.__dict__.update(outs)

#############################

class MidiPacket():
    def __init__(self, note=69, gate=1, vel=1, relvel=0):
        self.note=note
        self.gate=gate
        self.vel=vel
        self.relvel=relvel

class Keyboard(ModuleBuilder):
    def __init__(self, name='keyboard'):
        super().__init__(
            name=name,
            ins=['keyboard_in'], params=[],
            outs=['note', 'gate', 'vel', 'relvel'],
            function=self.parse_midi)

    def parse_midi(self, keyboard_in=MidiPacket()):
        return keyboard_in.__dict__

#############################

class Pulse1(ModuleBuilder):
    def __init__(self, name='pulse1'):
        self.current = 0
        self.launchflag = False
        super().__init__(
            name=name,
            ins=['Input'], params=['width'],
            outs=['Output'],
            function=self.pulse)

    def pulse(self, Input=1, width=0.1):
        launch = self.trigger(Input)
        if launch:
            self.launchflag = True
            self.clock = 0
        if not self.launchflag:
            return {'Output':0}
        x = self.clock / self.sr
        output = {'Output':waveform.pulse(x, width)}
        return output

    def trigger(self, val):
        current = self.current
        self.current = self.Input
        if val > 0 and current <= 0:
            return 1
        else:
            return 0

#############################

class Amplifier1(ModuleBuilder):
    def __init__(self, name='amplifier1'):
        super().__init__(
            name=name,
            ins=['Input'], params=['factor'],
            outs=['Output'],
            function=self.amplify)

    def amplify(self, Input=0, factor=1):
        output = {'Output':Input * factor}
        return output











































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


class WhiteNoiseGenerator(ModuleBuilder):
    def __init__(self, name='wnoise', maxamp=1):
        self.maxamp = maxamp
        self.sat = SimpleSaturation(maxamp=maxamp).function
        super().__init__(
            name=name, n_in=0, n_out=1, function=self.function
        )

    def function(self):
        return self.sat(random.gauss(0, 1))

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

class SimpleVibrato(ModuleBuilder):
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

###################

class SimpleFlanger(ModuleBuilder):
    '''a méthode additive, où le son retardé est ajouté tel quel,
    l'effet dépend du retard appliqué ;
    la méthode soustractive, où le son ajouté est
    l'inverse de la courbe du premier ;
    le flanger Through-Zero, où l'on applique aussi un
    retard au signal original pour que le signal modulé passe avant'''

    def __init__(self, name='flanger',
                 start=0, stop=50, freq=5, kind='add'):
        self.start = start
        self.stop = stop
        self.freq = freq
        self.lfo = lambda x: int(
            start
            + (stop - start)
                * waveform.Waveform(center=0.5, ampl=0.5).sin(x)
        )
        if kind is 'add':
            self.combine = lambda x, y: x + y
        elif kind is 'sub':
            self.combine = lambda x, y: x - y
        self._init_flag = False

        super().__init__(
            name=name, n_in=1, n_out=1, function=self.function
        )

    def function(self, x):
        if self._init_flag == False:
            self._init_queue()
        val = self.memo.prepend(x)
        index = self.lfo(self.clock / self.sr * 2 * np.pi * self.freq)
        return self.combine(x, self.memo.queue[index])

    def _init_queue(self):
        assert self.sr is not None, 'Must upload the sr first'
        length = int(self.stop+1)
        self.memo = NumpyQueue(maxsize=length)
        self._init_flag = True
