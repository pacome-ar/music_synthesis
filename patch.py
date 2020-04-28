import modules, oscillators, envelope, synth, Plugboard

def patch_with_2osc_1env():

    kbd = modules.Keyboard('kbd')
    oscA1 = oscillators.OscillatorA('oscA1')
    oscA2 = oscillators.OscillatorA('oscA2')
    mix = modules.Mixer3('mix')
    env = envelope.ADSR_envelope('env')
    mono = synths.MonophonicSynth()
    mono.add_modules(kbd, oscA1, oscA2, mix, env);

    pb = plugboard.Plugboard('kbd', 'env')
    pb.make_cable('kbd gate', 'env gate')
    pb.make_cable('oscA1 Output', 'mix Input1')
    pb.make_cable('oscA2 Output', 'mix Input2')
    pb.make_cable('mix Output', 'env Input')

    # set parameters:
    mono.modules['oscA1'].coarse_tuning = 64
    mono.modules['oscA1'].fine_tuning = 0
    mono.modules['oscA1'].waveform_selector = 'square'
    mono.modules['oscA1'].pulse_width = 1

    mono.modules['oscA2'].coarse_tuning = 64
    mono.modules['oscA2'].fine_tuning = 0
    mono.modules['oscA2'].waveform_selector = 'tri'
    mono.modules['oscA2'].pulse_width = 1

    mono.modules['mix'].mix1 = 1
    mono.modules['mix'].mix2 = 1

    mono.modules['env'].attack_type = 'log'
    mono.modules['env'].attack = 0.5e-3
    mono.modules['env'].decay = 520e-3
    mono.modules['env'].sustain = 1
    mono.modules['env'].release = 0.5e-3
    mono.modules['env'].gate_trigg_selector = 0
    mono.modules['env'].amp = 1
    mono.modules['env'].amp = alpha=3

    # make inputs
    kbd_in = [modules.MidiPacket(gate=j)
              for j in np.repeat([1], 1000)]

    # run synth
    mono._synchronize_clocks()
    mono._upload_sr()
    out = []
    for in_ in kbd_in:
        mono.modules['kbd'].keyboard_in = in_
        plugboard._plug_plugboard(mono, pb)
        mono.modules['mix']()
        plugboard._plug(mono.modules['mix'], 'Output',
                        mono.modules['env'], 'Input')
        plugboard._plug_plugboard(mono, pb)
        out.append(mono.modules[pb.OUT].Output)
        mono._advance_all_clocks()
