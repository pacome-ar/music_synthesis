import numpy as np
import modules


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

    def make_filter(self):
        if self.frequency == self.filter_freq:
            return
        else:
            self.filter_freq = self.frequency
            w = 2*np.pi * self.frequency / self.sr
            a = w / (1 + w)
            self.iir = Iir(ylist=[(1 - a)], xlist=[a])

    def filter_(self, Input=0, frequency=504):
        Input, frequency = self._get_default_value(
            [Input, frequency], [0, 504]
        )
        self.make_filter()
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

    def make_filter(self):
        if self.frequency == self.filter_freq:
            return
        else:
            self.filter_freq = self.frequency
            w = 2*np.pi * self.frequency / self.sr
            a = 1 / (1 + w)
            self.iir = Iir(ylist=[a], xlist=[a, -a])

    def filter_(self, Input=0, frequency=504):
        Input, frequency = self._get_default_value(
            [Input, frequency], [0, 504]
        )
        self.make_filter()
        Output = self.iir(Input)
        return {'Output':Output}
