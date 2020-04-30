import modules, plugboard
from collections import OrderedDict, defaultdict


class Network():
    def __init__(self):
        self.links = []
        self.starts = defaultdict(list)
        self.ends = defaultdict(list)
        self.nodes = dict()

    def add_link(self, start, end):
        self.links.append([start, end])
        self.starts[start].append(end)
        self.ends[end].append(start)
        self.nodes.update({start:-1, end:-1})

    def add_links(self, *links):
        for link in links:
            self.add_link(*link)

    def get_leaves(self):
        return list(set(self.starts.keys())- set(self.ends.keys()))

    def recurse_make_network(self, leaves, index):
        for leaf in leaves:
            self.nodes[leaf] = max(self.nodes[leaf], index)
            newleaves = self.starts[leaf]
            self.recurse_make_network(newleaves, index + 1)

    def make_network(self):
        self.recurse_make_network(self.get_leaves(), 0)

    def get_ordered_nodes(self):
        self.make_network()
        ordered_nodes = sorted(self.nodes, key=self.nodes.get)
        self.ordered_nodes = ordered_nodes
        return ordered_nodes


class Plugboard():
    def __init__(self):
        self.cables = []
        self.modules_start = defaultdict(list)
        self.modules_end = defaultdict(list)
        self.network = Network()

    def make_cable(self, start, end):
        idx = len(self.cables)
        start_mod, start_port = start
        end_mod, end_port = end
        self.modules_start[start_mod].append(idx)
        self.modules_end[end_mod].append(idx)
        self.cables.append(
            [(start_mod, start_port), (end_mod, end_port)]
        )
        self.network.add_link(start_mod, end_mod)

    def make_cables(self, *cables):
        for cable in cables:
            self.make_cable(*cable)


def plug(start_mod, start_port, end_mod, end_port):
    end_mod.__dict__[end_port] = start_mod.__dict__[start_port]


def run_synth(pb, modules):
    '''given a plugboard and a list of modules
    calls the modules and plug cable in the right order
    the network ordered node must be precalculated
    '''
    for mod in pb.network.ordered_nodes:
        cableidxs = pb.modules_end[mod]
        if len(cableidxs):
            for idx in cableidxs:
                cable = pb.cables[idx]
                (start_mod, start_port), (end_mod, end_port) = cable
                plug(modules[start_mod], start_port,
                     modules[end_mod], end_port)
        modules[mod]()


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
        self.pb = Plugboard()
        # self._upload_sr()
        # self._synchronize_clocks()

    def __add__(self, module):
        name = module.name
        module.sr = self.sr
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

    def run(self):
        '''compute the output of the synth'''
        run_synth(self.pb, self.modules)
