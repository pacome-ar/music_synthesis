import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg
from matplotlib import pyplot as plt

import modules

class Oscilloscope():
    def __init__(
        self, figsize=(4, 3), dpi=100,
        buffersize=200, ratio=10
    ):
        self.figsize = figsize
        self.dpi = dpi
        self.buffersize = buffersize
        self.ratio = ratio
        self.datas = [0] * buffersize
        self.counter = 0
        self._init_fig()

    def _init_fig(self):
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.gca()
        ax.set_ylim(-1.05, 1.05)
        self.line, = ax.plot(self.datas)
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()

        self.fig = fig
        self.raw_data = raw_data

    def _update_fig(self, datas):
        self.fig.canvas.flush_events()
        self.line.set_ydata(datas)
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        self.raw_data = raw_data

    def _tick(self):
        self.counter = (self.counter + 1) % self.ratio

    def _load(self, data):
        if self.counter == 0:
            self.datas = [data] + self.datas[:-1]
            self._update_fig(self.datas)
            return True
        return False

    def load_and_tick(self, datas):
        ret = self._load(datas)
        self._tick()
        return ret
