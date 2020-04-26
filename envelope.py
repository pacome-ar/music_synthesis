import numpy as np
import modules


def choose_attack_func(kind='lin'):
    if kind == 'lin':
        return attack_func_lin
    elif kind == 'log':
        return attack_func_log
    elif kind == 'exp':
        return attack_func_exp

def attack_func_lin(x, attack=1, amp=1, alpha=5):
    return amp * (x / attack)

def attack_func_exp(x, attack=1, amp=1, alpha=5):
    A = amp / (np.exp(alpha) - 1)
    return A * (np.exp(alpha * x / attack) - 1)

def attack_func_log(x, attack=1, amp=1, alpha=5):
    return amp * np.log(1 + alpha * x / attack) / np.log(1 + alpha)

def decay_func(x, decay=1, amp1=1, amp2=0, alpha=5):
    A = (amp2 - amp1) / (np.exp(-alpha) - 1)
    return A * (np.exp(-alpha * x / decay) - 1) + amp1

def make_ads(attack=1, decay=1, sustain=1,
         afunc=attack_func_log, alpha_attack=5, alpha_decay=5):
    def wrapper(x):
        if x < attack:
            return afunc(x, attack=attack)
        elif x <= attack + decay:
            return decay_func(x - attack, decay=decay, amp2=sustain)
        else:
            return sustain
    return wrapper


class AD_envelope(modules.ModuleBuilder):
    def __init__(self, name='ad_envelope1'):
        ins = ['gate_trigg', 'amp', 'Input']
        params = ['gate_trigg_selector',
                  'attack',
                  'decay']
        outs = ['envelope_out, Output']
        super().__init__(
            name=name,
            ins=ins, params=params,
            outs=outs,
            function=self.envelope
        )
        self.attack_flag = False
        self.decay_flag = False
        self.gate_flag = False
        self.trigger_flag = False
        self.former_attack = False
        self.decay_amp = 1
        self.amp = 1
        self.alpha = 5

    def envelope(
        self,
        gate_trigg=False,
        amp=1,
        Input=None,
        gate_trigg_selector=0,
        attack=0.5e-3,
        decay=0.5e-3
    ):
        alpha = self.alpha

        if not gate_trigg_selector:
            if gate_trigg > 0:
                if not self.gate_flag:
                    self._init_attack_phase()
                    envelope_out = self._get_attack(attack, amp)
                elif self.gate_flag:
                    envelope_out = self._choice(attack, decay, amp, alpha)
                self.gate_flag = True

            elif gate_trigg <= 0:
                if self.decay_flag:
                    envelope_out = self._get_decay(decay, alpha)
                elif not self.decay_flag:
                    if self.attack_flag:
                        self._init_decay_phase()
                        envelope_out = self._get_decay(decay, alpha)
                    elif not self.attack_flag:
                        envelope_out = 0
                self.gate_flag = False

        elif gate_trigg_selector:
            if gate_trigg > 0:
                if not self.trigger_flag:
                    self._init_attack_phase()
                    envelope_out = self._get_attack(attack, amp)
                elif self.trigger_flag:
                    envelope_out = self._choice(attack, decay, amp, alpha)
                self.trigger_flag = True

            if gate_trigg <= 0:
                envelope_out = self._choice(attack, decay, amp, alpha)
                self.trigger_flag = False

        if Input is not None:
            Output = Input * envelope_out
        else:
            Output = None

        return {'Output':Output, 'envelope_out':envelope_out}

    def _choice(self, attack, decay, amp, alpha):
        if not self.attack_flag and not self.decay_flag:
            return 0
        elif self.attack_flag:
            return self._get_attack(attack, amp)
        elif self.decay_flag:
            return self._get_decay(decay, alpha)

    def _init_decay_phase(self):
        self.clock = 0
        self.decay_flag = True
        self.attack_flag = False
        self.decay_amp = self.former_attack

    def _init_attack_phase(self):
        self.clock = 0
        self.attack_flag = True
        self.decay_flag = False

    def _get_attack(self, attack, amp):
        x = self.clock / self.sr
        self.former_attack = attack_func_lin(x, amp=amp, attack=attack)
        if x == attack:
            self._init_decay_phase()
        return self.former_attack

    def _get_decay(self, decay, alpha):
        x = self.clock / self.sr
        if x == decay:
            self.decay_flag = False
        return decay_func(
            x, amp1=self.decay_amp, decay=decay, alpha=alpha
        )


class ADSR_envelope(modules.ModuleBuilder):
    def __init__(self, name='adsr_envelope1'):
        ins = ['gate', 'retrig', 'amp', 'Input']
        params = ['attack', 'decay', 'sustain', 'release',
                  'attack_type', 'invert']
        outs = ['envelope_out, Output']
        super().__init__(
            name=name,
            ins=ins, params=params,
            outs=outs,
            function=self.envelope
        )
        self.current_ads_env = 0
        self.gate_flag = False
        self.trigg_flag = False
        self.release_flag = False

    def envelope(
        self,
        gate=0,
        amp=1,
        retrig=0,
        Input=None,
        attack=0.5e-3,
        decay=0.5e-3,
        sustain=0.6,
        release=0.5e-3,
        attack_type='log',
        invert=False
    ):
        if not retrig:
            retrig = 0
        alpha_attack = 5
        alpha_decay = 5
        alpha_release = 5
        attac_func = choose_attack_func(attack_type)
        self.ads = make_ads(
            attack=attack, decay=decay, sustain=sustain,
            afunc=attac_func,
            alpha_attack=alpha_attack, alpha_decay=alpha_decay
        )

        if gate > 0:
            if not self.gate_flag:  # gate changes from 0 to 1
                self._restart_clock()
                envelope_out = self._get_ads()
            elif self.gate_flag: # gate stays up
                if not self.trigg_flag and retrig > 0:
                    self._restart_clock()
                    envelope_out = self._get_ads()
                else:
                    envelope_out = self._get_ads()
            self.gate_flag = True
            self.current_ads_env = envelope_out

        elif gate <= 0:
            if self.gate_flag: # gate changes from 1 to 0
                self._init_release()
                envelope_out = self._get_release(
                    release, alpha_release)
            elif not self.gate_flag:
                if self.release_flag:
                    envelope_out = self._get_release(
                        release, alpha_release)
                elif not self.release_flag:
                    envelope_out = 0
            self.gate_flag = False

        self.trigg_flag = retrig > 0

        envelope_out *= amp

        if invert:
            envelope_out = -envelope_out

        if Input is not None:
            Output = Input * envelope_out
        else:
            Output = None

        return {'Output':Output, 'envelope_out':envelope_out}

    def _restart_clock(self):
        self.clock = 0

    def _init_release(self):
        self.clock = 0
        self.release_flag = True

    def _get_ads(self):
        x = self.clock / self.sr
        return self.ads(x)

    def _get_release(self, release, alpha_release):
        x = self.clock / self.sr
        if x == release:
            self.release_flag = False
        return decay_func(
            x,
            decay=release, amp1=self.current_ads_env,
            amp2=0, alpha=alpha_release
        )

######################################
################ TEST ################
######################################

import waveform

def test_ad_enveloppe(gate_trigg_selector=True,
                      sr=44000,
                      gate_freq = 0.8,
                      attack=0.1,
                      trigg=False,
                      alpha=5):

    x = np.arange(2*sr)
    gate = waveform.Waveform(sr=sr, sym=False).wrapp_func(
                waveform.square, x, gate_freq).astype(bool)

    if trigg:
        gate = gate * 0
        gate[::1000] = 1

    signal = waveform.Waveform(sr=sr, sym=False).wrapp_func(
                waveform.sin, x, 440)

    out_env = []
    out = []

    envelope = AD_envelope()
    envelope.sr = sr
    envelope.gate_trigg_selector = gate_trigg_selector
    envelope.attack = attack
    envelope.decay = 0.5
    envelope.alpha = alpha
    for s, g in zip(signal, gate):
        envelope.clock += 1
        envelope.gate_trigg = g
        envelope.Input = s
        envelope()
        out_env.append(envelope.envelope_out)
        out.append(envelope.Output)

    return gate, out, out_env


def test_adsr_enveloppe(sr=44000,
                      gate_freq = 0.8,
                      attack=0.1,
                      sustain=0.6,
                      release=0.6,
                      trigg=False,
                      attack_type='log',
                      invert=False):

    x = np.arange(2*sr)
    gate = [waveform.pulse(xx, width=gate_freq*sr) for xx in x]

    signal = waveform.Waveform(sr=sr, sym=False).wrapp_func(
                waveform.sin, x, 440)

    trigger = np.array(gate) * 0
    trigger[::int(sr/3)] = 1 * trigg

    out_env = []
    out = []

    envelope = ADSR_envelope()
    envelope.sr = sr
    envelope.attack = attack
    envelope.decay = 0.5
    envelope.sustain = sustain
    envelope.release = release
    envelope.amp = 1
    envelope.attack_type = attack_type
    envelope.invert = invert

    for s, g, t in zip(signal, gate, trigger):
        envelope.clock += 1
        envelope.gate = g
        envelope.retrig = t
        envelope.Input = s
        envelope()
        out_env.append(envelope.envelope_out)
        out.append(envelope.Output)

    return gate, out, out_env, trigger
