import numpy as np
import plugboard, modules, synths, midi, notes_and_modes

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

def octave(mode='major'):
    mono, music = make_single_note()
    mod = notes_and_modes.Modes().__dict__[mode]
    notes = np.arange(25)
    durs = np.ones(len(notes)) * 0.15
    amps = np.ones(len(durs))
    score = np.vstack([50 + mod._intervalToIndex(notes), durs, amps]).T
    inputs = midi.parse_midi(score, quarternote=1, tempo=150)
    print(midi.midi_note_map[score[:, 0].astype(int)])
    music = np.concatenate(
        [mono.play_note(note) for note in inputs]
    )
    return music

def auclairdelalune(mode='major'):
    mono, music = make_single_note()
    mod = notes_and_modes.Modes().__dict__[mode]
    notes = np.array([0, 0, 0, 1, 2, 1, 0, 2, 1, 1, 0,
                  0, 0, 0, 1, 2, 1, 0, 2, 1, 1, 0,
                  1, 1, 1, 1, -2, -2, 1, 0, -1, -2, -3,
                  0, 0, 0, 1, 2, 1, 0, 2, 1, 1, 0,
                 ])
    durs = np.array([0.5, 0.5, 0.5, 0.5, 1, 1, 0.5, 0.5, 0.5, 0.5, 2,
                    0.5, 0.5, 0.5, 0.5, 1, 1, 0.5, 0.5, 0.5, 0.5, 2,
                    0.5, 0.5, 0.5, 0.5, 1, 1, 0.5, 0.5, 0.5, 0.5, 2,
                    0.5, 0.5, 0.5, 0.5, 1, 1, 0.5, 0.5, 0.5, 0.5, 2])
    amps = np.ones(len(durs))
    score = np.vstack([60 + mod._intervalToIndex(notes), durs, amps]).T
    inputs = midi.parse_midi(score, quarternote=1, tempo=150)
    print(midi.midi_note_map[score[:, 0].astype(int)])

    music = np.concatenate(
        [mono.play_note(note) for note in inputs]
    )
    return music
