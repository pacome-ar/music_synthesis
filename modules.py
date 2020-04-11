import numpy as np
from scipy.interpolate import interp1d

def build_cst_function(val=1):
    def func(*x, **kwargs):
        return val
    return func

class ModuleBuilder():
    def __init__(self, name='', n_in=1, n_out=1, parameters=[],
                       function=build_cst_function(1)):
        self.name = name
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

class InputModule(ModuleBuilder):
    '''will be the only one communing with exterior
    should transform (freq, dur, amp) into signal'''
    def __init__(
        self, name='INPUT_module', wf='sin', sr=48000, unit='Hz'
    ):
        self._parse_sr(sr, unit)
        if wf is 'sin':
            wf = np.sin
        self.wf = wf
        super().__init__(
            name=name, n_in=1, n_out=1, function=self.translate_note)

    def _parse_sr(self, sr, unit):
        if unit == 'kHz':
            self.sr = sr * 1e3
        elif unit == 'Hz':
            self.sr = sr
        else:
            raise Exception('unit {} not understood'.format(unit))

    def translate_note(self, input_):
        freq, dur, amp = input_
        nbpoint = dur * self.sr
        return amp * self.wf(
                    np.arange(nbpoint) * 2 * np.pi / self.sr * freq)

###################

class SimpleEnvelope(ModuleBuilder):
    def __init__(self, name='smpl_env', *ys):
        if len(ys) == 0:
            ys = [1]
        self.ys = ys
        super().__init__(
            name=name, n_in=1, n_out=1, function=self.function
        )

    def _make_interpolant(self):
        ts = np.linspace(0, 1, len(self.ys) + 2)
        interp = interp1d(
            ts, np.concatenate(([0], self.ys, [0])), kind='cubic'
        )
        self.interp = interp

    def function(self, input_):
        self._make_interpolant()
        env = self.interp(np.linspace(0, 1, len(input_)))
        return env * input_

####################

class SimpleSaturation(ModuleBuilder):
    def __init__(self, name='simpl_sat', maxamp=1):
        self.maxamp = maxamp
        super().__init__(
            name=name, n_in=1, n_out=1, function=self.saturate
        )

    def saturate(self, input_):
        input_ = np.asarray(input_)
        target = input_.max() * self.maxamp
        input_[input_ > target] = target
        input_[input_ < -target] = -target
        return input_
