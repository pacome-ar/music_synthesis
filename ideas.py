import numpy as np

###########################

def step_sample(x, nb=2):
    x = np.array(x)
    N = len(x)
    x = np.repeat(x[::nb], nb)[:N]
    return x

from scipy.interpolate import interp1d

###########################

def pitch(x, fact=1.001, keepsize=True):
    N = len(x)
    old_smpling = np.arange(N)
    interp = interp1d(old_smpling, x)
    new_smpling = np.linspace(0, N-1, int(N * fact), endpoint=True)
    if keepsize:
        Np = len(new_smpling)
        tmp = np.zeros(N)
        tmp[:min(N, Np)] = new_smpling[:min(N, Np)]
        new_smpling = tmp

    return interp(new_smpling)

###########################

def make_lfo(freq, func=np.sin, amp=1, sr=48000):
    fact = 2 * np.pi / sr * freq
    def wrapper(N):
        return amp * func(fact * np.arange(N))
    return wrapper

###########################

def expo(x, decay):
    if isinstance(decay, (int, float)):
        decay = np.ones_like(x) * decay
    assert len(decay) == len(x)
    fact = np.ones_like(x)
    fact[decay < 0] = 1 / np.exp(np.abs(decay[decay < 0]))
    return fact * np.exp(- decay * x)

###########################

def prepare_steps(x, step=0.1, center=0):
    xmin, xmax = x.min(), x.max()
    ret = np.concatenate(
            (np.arange(
                    center - 1 - (1 * bool(center%step)
                              + (center - xmin) // step) * step,
                    center,
                    step),
             np.arange(center, xmax+step, step)
            )
        )
    return ret

def quantize(x, steps):
    steps = np.asarray(list(steps))
    x = np.asarray(x)
    msk = x[:, None] < steps
    maxs = np.argmax(msk, axis=1)
    newx = steps[maxs - 1]
    return newx

def make_quantize(step=1, start=0, amp=1):
    def wrapper(x):
        steps = np.arange(start, x.max()+step, step)
        return amp * quantize(x, steps)
    return wrapper






















################################

# Add multiplier / adder modules !

















################################
