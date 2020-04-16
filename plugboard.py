from collections import defaultdict

class Plugboard():
    def __init__(self, IN, OUT, *plugs):
        self.IN = IN
        self.OUT = OUT
        self.cables = []
        self.modules_ports_in = defaultdict(list)
        self.modules_ports_out = defaultdict(list)
        [self.make_cable(*plug) for plug in plugs]

    def make_cable(self, in_, out_):
        futur_idx = len(self.cables)
        in_mod, in_port = self._parse_entry(in_)
        out_mod, out_port = self._parse_entry(out_)
        self.modules_ports_in[in_mod].append(futur_idx)
        self.modules_ports_out[out_mod].append(futur_idx)
        self.cables.append(((in_mod, in_port), (out_mod, out_port)))

    def _parse_entry(self, entry):
        module, port = entry.split(' ')
        return module, port

    def get_modules_without_input_cables(self):
        # something goes out but nothing goes in and not IN
        module_with_input_cables = set(self.modules_ports_in.keys())
        module_with_output_cables = set(self.modules_ports_out.keys())
        return list(module_with_input_cables
                  - module_with_output_cables
                  - set(self.IN))

##################################

def make_cable_from_list(names):
    IN = names[0]
    OUT = names[-1]
    cables = []
    for (n1, n2) in zip(names[:-1], names[1:]):
        cables.append((n1 + ' output_1', n2 + ' input_1'))
    return IN, OUT, cables

##################################

class LinearPlugboard(Plugboard):
    def __init__(self, names):
        IN, OUT, plugs = make_cable_from_list(names)
        super().__init__(IN, OUT, *plugs)

def _plug(a, in_, b, out_):
    a()
    b.__dict__[out_] = a.__dict__[in_]
    # print('pluging ', a.name, in_, ' on ', b.name, out_)

##################################

def recurse_plug(module, synth, plugboard):
    cables_id = plugboard.modules_ports_in[module]
    modouts = []
    for cid in cables_id:
        cable = plugboard.cables[cid]
        (modin, portin), (modout, portout) = cable
        _plug(
            synth.modules[modin], portin,
            synth.modules[modout], portout
        )
        modouts.append(modout)
    for modout in modouts:
        recurse_plug(modout, synth, plugboard)

def _plug_plugboard(synth, pb):
    synth.modules[pb.IN]()
    [synth.modules[mod]() for mod in pb.get_modules_without_input_cables()]
    recurse_plug(pb.IN, synth, pb)
    synth.modules[pb.OUT]()





###########################
# Workflow of the plugboard (warning! doesn work with loops)

# 1) plug the modules that don't have inputs
# 2) plug the starting module and port
# 3) plug all the cables that have the starting module as from modules
# 4) for all these plugs, plug the attached modules recursively

### recursion:

# start with module
#   get all cables with this module as entry (input)
#   plug all these cables
#   get all the modules which are outputs of these cables
#   for all these modules, run recursion
