import arbor
import arbor_pycat

@arbor_pycat.register
class RandTest(arbor_pycat.CustomMechanism):
    random = [('rand', 0)]
    name = 'rt'
    def init_mechanism(self, pp):
        pass
    def compute_currents(self, pp):
        pp.i = 0
        pp.v = pp.rand

cat = arbor_pycat.build()

# import matplotlib.pyplot as plt

tree = arbor.segment_tree()
tree.append(arbor.mnpos, arbor.mpoint(-3, 0, 0, 3), arbor.mpoint(3, 0, 0, 3), tag=1)

# (2) Define the soma and its midpoint
labels = arbor.label_dict({"soma": "(tag 1)", "midpoint": "(location 0 0.5)"})

# (3) Create cell and set properties
decor = (
    arbor.decor()
    .set_property(Vm=-40)
    .paint('"soma"', arbor.density("rt"))
)

class single_recipe(arbor.recipe):
    def __init__(self):
        arbor.recipe.__init__(self)
        self.the_props = arbor.neuron_cable_properties()
        self.the_props.catalogue.extend(cat, '')
    def num_cells(self): return 1
    def cell_kind(self, _): return arbor.cell_kind.cable
    def cell_description(self, gid): return arbor.cable_cell(tree, decor, labels)
    def probes(self, _): return [arbor.cable_probe_membrane_voltage('(root)'),]
    def global_properties(self, kind): return self.the_props
recipe = single_recipe()
sim = arbor.simulation(recipe)
handle = sim.sample((0, 0), arbor.regular_schedule(0.1))
sim.run(tfinal=1000)
data, meta = sim.samples(handle)[0]
v = data[:, 1]
assert abs(v.mean()) < 0.05
assert abs(v.std() - 1) < 0.05
