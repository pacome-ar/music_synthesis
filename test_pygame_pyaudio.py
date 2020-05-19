import pygame
import time
import pyaudio
import numpy as np

import oscillators
import gui_tools

################
# pygame params
display_width = 800
display_height = 600
event_FPS = 50 # Frames per second.
display_FPS = 10 # Frames per second.
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

################
# pyaudio setup
sr = 44100
frames_per_buffer = 1024
# server_queue_size = frames_per_buffer
#
# server_queue = [0] * server_queue_size  # global buffer

# def callback(in_data, frame_count, time_info, status):
#     data = server_queue[-frame_count:]
#     tone_out = np.array(data).astype(np.float32).tostring()
#     return (tone_out, pyaudio.paContinue)

################
# music preparation
keys = [
    'K_c', 'K_f', 'K_v', 'K_g', 'K_b', 'K_n',
    'K_j', 'K_COMMA', 'K_k', 'K_SEMICOLON',
    'K_l', 'K_COLON', 'K_EXCLAIM'
    ]
notes = 'C C+ D D+ E F F+ G G+ A A+ B C'.split(' ')
midi_dict = dict(zip(keys, range(60, 73)))
note_dict = dict(zip(range(60, 73), notes))

osc = oscillators.OscillatorA()
osc.sr = sr
# osc.waveform_selector = 'tan'
################
# init pygame
successes, failures = pygame.init()
print("{0} successes and {1} failures".format(successes, failures))

screen = pygame.display.set_mode((display_width, display_height))
clock = pygame.time.Clock()

# GUI loop
note, mute = 69, True

############
# call back
def callback(in_data, frame_count, time_info, status):
    global osc
    tmp = []
    for i in range(frame_count+1):
        osc()
        osc.clock += 1
        tmp.append(osc.Output)
    tone_out = np.array(tmp).astype(np.float32).tostring()
    return (tone_out, pyaudio.paContinue)


# start audio stream
pya = pyaudio.PyAudio()
stream = pya.open(format=pyaudio.paFloat32,
                channels=1,
                rate=sr,
                output=True,
                stream_callback=callback,
                frames_per_buffer=frames_per_buffer)

# while stream.is_active():
#     pass

while True:
    # clocks ticking
    clock.tick(event_FPS)

    # detect events
    try:
        note, mute = gui_tools.detect_note_events(midi_dict, note, mute)
    except TypeError:
        stream.stop_stream()
        stream.close()
        pya.terminate()
        quit()

    # update synth with event detection
    osc.__dict__.update({'coarse_tuning':note, 'mute':mute})

    # sound loop
    # for i in range(int(sr / event_FPS)):
    #     osc()
    #     osc.clock += 1
    #     out = osc.Output
    #     server_queue = [out] + server_queue[:-1]
    # print(out)

    # display
    screen.fill(BLACK)
    gui_tools.message_display(
        screen, note_dict[note],
        (display_width/2), (display_height/4), textsize=50
        )

    pygame.display.update()  # Or pygame.display.flip()
