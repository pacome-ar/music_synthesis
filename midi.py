import numpy as np
import notes_and_modes

midi_note_map = [None] * 127
midi_note_map[21:] = np.concatenate([notes_and_modes.make_octave(num=i)
                         for i in range(9)])
midi_note_map = np.array(midi_note_map)
note_midi_map = dict(list(zip(midi_note_map[21:], range(21, 129))))

def note_to_freq(note, ref=440):
    return ref * 2**((note - 69) / 12)

def freq_to_note(freq, ref=440):
    return np.log2(freq/ref) * 12 + 69

def note_value_to_sec(value, tempo=120, quarternote=0.25):
    return 60 / tempo / quarternote * value

def normalize_amplitude(amp, norm=1):
    if amp.min() == amp.max():
        return amp
    return (amp - amp.min()) / (amp.max() - amp.min()) * norm

def parse_midi(score, ref=440, tempo=120, norm=1, quarternote=0.25):
    # score is list of tuple-like with (note, duration amplitude)
    score = np.asarray(score)
    freqences = note_to_freq(score[:,0], ref)
    durations = note_value_to_sec(score[:,1], tempo, quarternote)
    amplitudes = normalize_amplitude(score[:,2], norm)
    return np.vstack((freqences, durations, amplitudes)).T
