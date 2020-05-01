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
            norm = (1 + 2*Q/w + Q*w/2)
            xlist = [1/norm, 0, -1/norm]
            ylist = [(Q*w - 4*Q/w) / norm,
                     (4*Q/w + Q*w/4 - 1) / norm]
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
