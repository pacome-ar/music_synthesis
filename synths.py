import modules, plugboard
from collections import OrderedDict

class MonophonicSynth():
    def _return_self(self):
        return self

    def copy(self, name=None):
        copied = copy.deepcopy(self)
        if name:
            copied.name = name
        return copied

    def __init__(self, name='mono', sr=44000):
        self.name = name
        self.modules = OrderedDict()
        self.clock = 0
        self.sr = sr
        # self._upload_sr()
        # self._synchronize_clocks()

    def __add__(self, module):
        name = module.name
        module.sr = self.sr
        module.initialize()
        self.modules[name] = module
        return self

    def remove_modules(self, *mods):
        for mod in mods:
            del self.modules[mod]
        return self

    def add_modules(self, *mods):
        for mod in mods:
            self += mod
        return self

    def _synchronize_clocks(self):
        for module in self.modules.values():
            module.clock = self.clock

    def _upload_sr(self):
        for module in self.modules.values():
            module.sr = self.sr

    def _advance_clock(self):
        self.clock += 1

    def _advance_all_clocks(self):
        self._advance_clock()
        for module in self.modules.values():
            module.clock += 1

    #
    # def play_note(self, note):
    #     self.modules[self.pb.IN].input_1 = note
    #     plugboard._plug_plugboard(self, self.pb)
    #     return self.modules[self.pb.OUT].output_1
