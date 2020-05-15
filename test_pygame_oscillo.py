import pygame
import time
import oscillators
import gui_tools

successes, failures = pygame.init()
print("{0} successes and {1} failures".format(successes, failures))


display_width = 800
display_height = 600

screen = pygame.display.set_mode((display_width, display_height))
clock = pygame.time.Clock()
FPS = 1000 # Frames per second.

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
# RED = (255, 0, 0), GREEN = (0, 255, 0), BLUE = (0, 0, 255).

# rect = pygame.Rect((0, 0), (32, 32))
# image = pygame.Surface((32, 32))
# image.fill(WHITE)

def text_objects(text, font):
    textSurface = font.render(text, True, WHITE)
    return textSurface, textSurface.get_rect()

def message_display(text):
    largeText = pygame.font.Font('freesansbold.ttf',50)
    TextSurf, TextRect = text_objects(text, largeText)
    TextRect.center = ((display_width/2),(display_height/4))
    screen.blit(TextSurf, TextRect)

keys = [
    'K_c', 'K_f', 'K_v', 'K_g', 'K_b', 'K_n',
    'K_j', 'K_COMMA', 'K_k', 'K_SEMICOLON',
    'K_l', 'K_COLON', 'K_m', 'K_EXCLAIM'
    ]
notes = 'C C+ D D+ E F F+ G G+ A A+ B'.split(' ')
midi_dict = dict(zip(keys, range(60, 72)))
note_dict = dict(zip(keys, notes))

screen.fill(BLACK)
# screen.blit(image, rect)
note = 0
mute = True
osc = oscillators.OscillatorC()
osc.sr = FPS

oscillo = gui_tools.Oscilloscope(
    figsize=(4, 3), dpi=110,
    buffersize=200, ratio=10)

subscreen = pygame.display.get_surface()
size = oscillo.fig.canvas.get_width_height()
counter = 0
refresh_fps = 3

refresh_rate = int(FPS / refresh_fps)

while True:
    t0 = time.time()
    clock.tick(FPS)
    osc.clock += 1
    counter = (counter + 1) % refresh_rate
    print(counter)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit()
        key = pygame.key.get_pressed()
        for k, v in midi_dict.items():
            if key[pygame.__dict__[k]]:
                screen.fill(BLACK)
                note = v
                mute = False
                message_display('pressed ' + str(note_dict[k]))
            if event.type == pygame.KEYUP:
                if event.key == pygame.__dict__[k]:
                    note = v
                    mute = True
                    screen.fill(BLACK)
                    message_display('released ' + str(note_dict[k]))
    #
    out = osc.oscillator(coarse_tuning=note, mute=mute)['Output']
    plot = oscillo.load_and_tick(out)
    if plot:
        surf = pygame.image.fromstring(oscillo.raw_data, size, "RGB")
        screen.blit(surf, ((display_width/4),(display_height/3)))
    # screen.fill(BLACK)
    # message_display(str(out))
    #
    pygame.display.update()  # Or pygame.display.flip()
    print(time.time() - t0)
