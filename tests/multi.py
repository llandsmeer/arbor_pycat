import numpy as np
import arbor
import arbor_pycat

try:
    from arbor import units as U
    mV = U.mV
except ImportError:
    mV = 1

@arbor_pycat.register
class Passive(arbor_pycat.CustomMechanism):
    name = 'passive'
    parameters = [('gid', '()', -1)]
    def init_mechanism(self, pp):
        print(pp.width, 'Vm', pp.v[pp.node_index])
    def compute_currents(self, pp):
        print('#', pp.gid, pp.v[pp.node_index], pp.node_index)
        pp.i[pp.node_index] = (pp.v[pp.node_index] - pp.gid) * 1e-2

cat = arbor_pycat.build()

# import matplotlib.pyplot as plt

tree = arbor.segment_tree()
tree.append(arbor.mnpos, arbor.mpoint(-3, 0, 0, 3), arbor.mpoint(3, 0, 0, 3), tag=1)
tree.append(arbor.mnpos, arbor.mpoint(-3, 0, 0, 3), arbor.mpoint(3, 0, 0, 3), tag=2)
tree.append(arbor.mnpos, arbor.mpoint(-3, 0, 0, 3), arbor.mpoint(3, 0, 0, 3), tag=2)

# (2) Define the soma and its midpoint
labels = arbor.label_dict({"soma": "(tag 1)", "dend": "(tag 2)", "midpoint": "(location 0 0.5)"})

# (3) Create cell and set properties

class single_recipe(arbor.recipe):
    def __init__(self):
        arbor.recipe.__init__(self)
        self.the_props = arbor.neuron_cable_properties()
        self.the_props.catalogue.extend(cat, '')
    def num_cells(self): return 4
    def cell_kind(self, _): return arbor.cell_kind.cable
    def cell_description(self, gid):
        decor = arbor.decor().set_property(Vm=2*mV).paint('"dend"', arbor.density('passive', dict(gid=1e-10 + gid)))
        return arbor.cable_cell(tree, decor, labels)
    def probes(self, _): return [arbor.cable_probe_membrane_voltage('(root)')]
    def global_properties(self, kind): return self.the_props
recipe = single_recipe()
sim = arbor.simulation(recipe)
handles = [sim.sample((i, 0), arbor.regular_schedule(0.1)) for i in range(recipe.num_cells())]
sim.run(tfinal=30)
v = np.array([sim.samples(handle)[0][0][:,1] for handle in handles]).T


print(v[-1])
print(np.round(v[-1], 1))
assert all(np.round(v[-1], 1) == np.arange(recipe.num_cells()))
