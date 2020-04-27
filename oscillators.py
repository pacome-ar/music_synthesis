import numpy as np
import waveform, midi, modules


class Master_OSC(modules.ModuleBuilder):
    def __init__(self, name='master_osc1'):
        ins = ['pitch_mod1', 'pitch_mod2']
        params = ['coarse_tuning', 'fine_tuning',
                  'pitch_mod_amp1', 'pitch_mod_amp2']
        outs = ['slv_out']
        super().__init__(
            name=name,
            ins=ins, params=params,
            outs=outs,
            function=self.oscillator
        )

    def oscillator(
        self,
        pitch_mod1=0,
        pitch_mod2=0,
        coarse_tuning=69,
        fine_tuning=0,
        pitch_mod_amp1=0,
        pitch_mod_amp2=0
    ):
        if not coarse_tuning:
            coarse_tuning = 69
        if not fine_tuning:
            fine_tuning = 0
        if not pitch_mod_amp1:
            pitch_mod_amp1 = 0
        if not pitch_mod_amp2:
            pitch_mod_amp2 = 0

        assert 1 <= coarse_tuning <= 128, ''\
                 'coarse tuning should be between 1 and 128'
        assert -1 <= fine_tuning <= 1, ''\
                 'coarse tuning should be between -1 and 1'

        freq = midi.note_to_freq(coarse_tuning)
        freq *= 2**(fine_tuning/12)

        mod = 0
        if pitch_mod1:
            mod += pitch_mod1 * pitch_mod_amp1
        if pitch_mod2:
            mod += pitch_mod2 * pitch_mod_amp2

        slv_out = freq * 2**(mod/12)

        return {'slv_out':slv_out}


class OscillatorC(modules.ModuleBuilder):
    def __init__(self, name='oscC1'):
        ins = ['pitch_mod', 'FMA', 'AM']
        params = ['coarse_tuning', 'fine_tuning',
                  'pitch_mod_amp', 'FMA_amp', 'mute']
        outs = ['slv_out', 'Output']
        super().__init__(
            name=name,
            ins=ins, params=params,
            outs=outs,
            function=self.oscillator
        )

    def oscillator(
        self,
        pitch_mod=0,
        FMA=0,
        AM=0,
        coarse_tuning=69,
        fine_tuning=0,
        pitch_mod_amp=0,
        FMA_amp=0,
        mute=False
    ):
        if not coarse_tuning:
            coarse_tuning = 69
        if not fine_tuning:
            fine_tuning = 0
        if not pitch_mod_amp:
            pitch_mod_amp = 0
        if not FMA_amp:
            FMA_amp = 0
        if not mute:
            mute = False

        assert 1 <= coarse_tuning <= 128, ''\
                 'coarse tuning should be between 1 and 128'
        assert -1 <= fine_tuning <= 1, ''\
                 'coarse tuning should be between -1 and 1'

        freq = midi.note_to_freq(coarse_tuning)
        freq *= 2**(fine_tuning/12)

        amp = 1
        if AM is not None:
            amp *= AM

        mod_lin = 0
        if FMA:
            mod_lin += FMA_amp * FMA

        mod_exp = 0
        if pitch_mod:
            mod_exp += pitch_mod * pitch_mod_amp

        slv_out = freq * 2**(mod_exp/12) + mod_lin

        Output = self._sine(slv_out)
        if mute:
            Output = 0

        return {'slv_out':slv_out, 'Output':Output}

    def _sine(self, freq):
        x = self.clock / self.sr
        return waveform.sin(x * (2 * np.pi) * freq)


######################################
################ TEST ################
######################################


def test_oscillatorc(
    pitch_mod_amp=0.05, pitch_mod_freq=50, sr=44000
):
    x = np.arange(4*sr)

    osc1 = OscillatorC()

    signal1 = waveform.Waveform(sr=sr, sym=True).wrapp_func(
                    waveform.sin, x, pitch_mod_freq)
    signal2 = waveform.Waveform(sr=sr, sym=False).wrapp_func(
                    waveform.square, x, 5)

    out = []
    out_slv = []

    osc1.pitch_mod_amp = pitch_mod_amp
    osc1.FMA_amp = 0

    for s1, s2 in zip(signal1, signal2):
        osc1.clock += 1
        osc1.pitch_mod = s1
        osc1.FMA = None
        osc1()
        out.append(osc1.Output)
        out_slv.append(osc1.slv_out)

    return out, out_slv
