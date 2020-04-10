import numpy as np
import plugboard, modules, synths

def test_inverse_functions():
    func1 = lambda x: x**2
    func2 = lambda x: np.sqrt(x)

    modsin = modules.ModuleBuilder('modsin', 1, 1, function=func1)
    modarc = modules.ModuleBuilder('modarc', 1, 1, function=func2)

    mono = synths.MonophonicSynth('mono', mods=[modsin, modarc])

    pb = plugboard.Plugboard(
        'modsin', 'modarc', ('modsin output_1', 'modarc input_1')
    )

    ###
    mono.modules['modsin'].input_1 = np.linspace(0, 10*np.pi, 100)

    plugboard._plug_plugboard(mono, pb)

    assert np.allclose(mono.modules['modsin'].input_1,
            mono.modules['modarc'].output_1), 'Test failed'

    print('Test inverse passed')

def make_single_note():
    inputmod = modules.InputModule(name='inputmod', sr=48000, unit='Hz')
    pb = plugboard.Plugboard('inputmod', 'inputmod')
    mono = synths.MonophonicSynth('mono', mods=[inputmod])
    mono.pb=pb

    note = (440., 1, 1)

    return mono, mono.play_note(note)
