import numpy as np

# class Waveform():
#     def __init__(self):
#         self.sin = np.sin
#         self.saw = lambda x : ((x - np.pi)%(2 * np.pi)) / (np.pi) - 1
#         self.square = lambda x : ((x - np.pi)%(2 * np.pi) // np.pi - 0.5) * 2
#         self.tri = lambda x: 2 / (np.pi) * np.abs((x - np.pi/2)%(2*np.pi) - np.pi) - 1

def sin_to_rect(n=1, amp=1):
    def wrapper(x):
        if n==0:
            return x * 0
        return ((4 * amp) / np.pi *
                np.array([np.sin((2*p + 1) * x) / (2*p + 1)
                          for p in range(n)]).sum(axis=0))
    return wrapper

def sin_to_tri(n=1, amp=1):
    def wrapper(x):
        if n==0:
            return x * 0
        return ((-8 * amp) / np.pi**2 *
                np.array([np.cos((2*p + 1) * x) / (2*p + 1)**2
                          for p in range(n)]).sum(axis=0))
    return wrapper
