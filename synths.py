import modules, plugboard

class MonophonicSynth():
    def _return_self(self):
        return self

    def copy(self, name=None):
        copied = copy.deepcopy(self)
        if name:
            copied.name = name
        return copied

    def __init__(self, name, mods=[], pb=None):
        self.name = name
        self.pb = pb
        self.modules = {}
        self.add_modules(*mods)

    def __add__(self, module):
        name = module.name
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

    def play_note(self, note):
        self.modules[self.pb.IN].input_1 = note
        plugboard._plug_plugboard(self, self.pb)
        return self.modules[self.pb.OUT].output_1
