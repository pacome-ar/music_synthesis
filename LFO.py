import numpy as np
import waveform, modules


class LFOA(modules.ModuleBuilder):
    def __init__(self, name='lfoa1'):
        ins = ['restart', 'rate_mod_in']
        params = ['rate', 'mono_switch', 'Hi_Lo_Sub_selector',
                  'rate_mod_amp', 'phase', 'waveform_selector',
                  'mute']
        outs = ['slv_out', 'Output']
        super().__init__(
            name=name,
            ins=ins, params=params,
            outs=outs,
            function=self.oscillator
        )
        self.restart_flag = False

    def oscillator(
        self,
        restart=0,
        rate_mod_in=0,
        rate=0,
        mono_switch=False,
        Hi_Lo_Sub_selector='Lo',
        rate_mod_amp=0,
        phase=0,
        waveform_selector='sin',
        mute=False
    ):

        (restart, rate_mod_in, rate,
         mono_switch, Hi_Lo_Sub_selector,
         rate_mod_amp, phase,
         waveform_selector, mute
        ) = self._get_default_value(
            [restart, rate_mod_in, rate,
             mono_switch, Hi_Lo_Sub_selector,
             rate_mod_amp, phase,
             waveform_selector, mute],
            [0, 0, 0, False, 'Lo', 0, 0, 'sin', False])

        self._assert_range(phase, 'phase',-1, 1)
        self._assert_range(rate, 'rate', -1, 1)

        # get frequency
        freq = self._make_freq(
            Hi_Lo_Sub_selector, rate,
            rate_mod_in, rate_mod_amp
        )

        # restart clock if needed
        if restart > 0 and not self.restart_flag:
            self.clock = 0
        self.restart_flag = restart > 0

        # make wave
        Output = self._calculate_wave(
            freq, phase, self._get_wf(waveform_selector)
        )

        if mute:
            Output = 0

        return {'slv_out':freq, 'Output':Output}

    def _calculate_wave(self, freq, phase, wf):
        phase = self._rescale(phase, -np.pi, np.pi)
        x = self.clock / self.sr
        return wf(x*2*np.pi*freq + phase)

    def _get_wf(self, waveform_selector):
        if waveform_selector is 'sin':
            return waveform.sin
        elif waveform_selector is 'square':
            return waveform.square
        elif waveform_selector is 'tri':
            return waveform.tri
        elif waveform_selector is 'saw':
            return waveform.saw
        elif waveform_selector is 'saw_desc':
            return waveform.saw_desc

    def _make_freq(self, Hi_Lo_Sub_selector,
                   rate, rate_mod_in, rate_mod_amp):

        if Hi_Lo_Sub_selector is 'Hi':
            fmin, fmax = 0.26, 392
        elif Hi_Lo_Sub_selector is 'Lo':
            fmin, fmax = 0.02, 24.4
        elif Hi_Lo_Sub_selector is 'Sub':
            fmin, fmax = 1/699, 1/5.46

        rate += rate_mod_in * rate_mod_amp

        if rate < -1:
            rate = -1
        if rate > 1:
            rate = 1

        # The frequency modulation rate is log scaled
        # freq = self._rescale(rate, np.log(fmin), np.log(fmax))
        # return np.exp(freq)
        freq = self._rescale(rate, fmin, fmax)
        return freq

    def _rescale(self, x, xmin, xmax):
        return (x + 1) * (xmax - xmin) / 2 + xmin


class LFOslvA(modules.ModuleBuilder):
    def __init__(self, name='lfoslva1'):
        ins = ['restart', 'master']
        params = ['rate', 'mono_switch',
                  'phase', 'waveform_selector',
                  'mute']
        outs = ['Output']
        super().__init__(
            name=name,
            ins=ins, params=params,
            outs=outs,
            function=self.oscillator
        )
        self.restart_flag = False

    def oscillator(
        self,
        restart=0,
        master=0,
        rate=1,
        mono_switch=False,
        phase=0,
        waveform_selector='sin',
        mute=False
    ):

        (restart, master, rate,
         mono_switch, phase,
         waveform_selector, mute
        ) = self._get_default_value(
            [restart, master, rate,
         mono_switch, phase,
         waveform_selector, mute],
            [0, 0, 1, False, 0, 'sin', False])

        self._assert_range(phase, 'phase',-1, 1)

        # get frequency
        freq = self._make_freq(master, rate)

        # restart clock if needed
        if restart > 0 and not self.restart_flag:
            self.clock = 0
        self.restart_flag = restart > 0

        # make wave
        Output = self._calculate_wave(
            freq, phase, self._get_wf(waveform_selector)
        )

        if mute:
            Output = 0

        return {'Output':Output}

    def _get_wf(self, waveform_selector):
        if waveform_selector is 'sin':
            return waveform.sin
        elif waveform_selector is 'square':
            return waveform.square
        elif waveform_selector is 'tri':
            return waveform.tri
        elif waveform_selector is 'saw':
            return waveform.saw
        elif waveform_selector is 'saw_desc':
            return waveform.saw_desc

    def _calculate_wave(self, freq, phase, wf):
        phase = self._rescale(phase, -np.pi, np.pi)
        x = self.clock / self.sr
        return wf(x*2*np.pi*freq + phase)

    def _make_freq(self, master, rate):
        freq = master*rate
        return freq

    def _rescale(self, x, xmin, xmax):
        return (x + 1) * (xmax - xmin) / 2 + xmin


######################################
################ TEST ################
######################################


def test_lfoa(
    rate_mod_amp=0.05, rate_mod_freq=50, sr=44000, rate=0
):

    x = np.arange(sr*2)

    osc1 = LFOA()

    signal1 = waveform.Waveform(sr=sr, sym=True).wrapp_func(
                    waveform.sin, x, rate_mod_freq)

    out = []
    out_slv = []

    osc1.sr = sr
    osc1.rate = rate
    osc1.rate_mod_amp = rate_mod_amp
    osc1.Hi_Lo_Sub_selector = 'Lo'
    for s1, s2 in zip(signal1, signal1):
        osc1.clock += 1
        osc1.rate_mod_in = s1
        osc1()
        out.append(osc1.Output)
        out_slv.append(osc1.slv_out)

    return out, out_slv
