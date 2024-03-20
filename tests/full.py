import arbor
import numpy as np
import arbor_pycat._core as acm

E = acm.add_global("e", "mV", -70)
X = acm.add_state("x", "mV", -90)
Y = acm.add_state("Y", "mV", -80)
B = acm.add_ion("Na")

def init_mechanism(pp):
    assert np.all(np.diff(pp.node_index) == 1)
    pp.state(X)[:] = 10

def advance_state(pp):
    pp.state(X)[:] += (pp.state(Y) + pp.glob(E) - pp.state(X))*pp.dt

def compute_current(pp):
    print(pp.node_index)
    pp.i[:] = pp.v/10

def write_ions(pp):
    idx = pp.ions(0).index
    erev = pp.ions(0).reversal_potential[idx]

acm.set_init(init_mechanism)
acm.set_advance_state(advance_state)
acm.set_compute_currents(compute_current)
acm.set_write_ions(write_ions)

so_name = acm.get_so_name()
cat = arbor.load_catalogue(so_name)
mech = cat['mech']

tree = arbor.segment_tree()
tree.append(arbor.mnpos, arbor.mpoint(-3, 0, 0, 3), arbor.mpoint(3, 0, 0, 3), tag=1)
tree.append(arbor.mnpos, arbor.mpoint(-3, 0, 0, 3), arbor.mpoint(3, 0, 0, 3), tag=1)
tree.append(arbor.mnpos, arbor.mpoint(-3, 0, 0, 3), arbor.mpoint(3, 0, 0, 3), tag=1)

# (2) Define the soma and its midpoint
labels = arbor.label_dict({"soma": "(tag 1)", "midpoint": "(location 0 0.5)"})

# (3) Create cell and set properties
decor = (
    arbor.decor()
    .set_property(Vm=-40)
    .paint('"soma"', arbor.density("mech"))
    .set_ion("Na", int_con=54.4, ext_con=2.5, rev_pot=-77)
)

class single_recipe(arbor.recipe):
    def __init__(self):
        arbor.recipe.__init__(self)
        self.the_props = arbor.neuron_cable_properties()
        self.the_props.catalogue.extend(cat, '')
        self.the_props.set_ion(
            ion="Na", int_con=54.4 , ext_con=2.5, rev_pot=-77, valence=1
        )
    def num_cells(self): return 1
    def cell_kind(self, _): return arbor.cell_kind.cable
    def cell_description(self, gid): return arbor.cable_cell(tree, decor, labels)
    def probes(self, _): return [arbor.cable_probe_membrane_voltage('(root)')]
    def global_properties(self, kind): return self.the_props
recipe = single_recipe()
sim = arbor.simulation(recipe)
handle = sim.sample((0, 0), arbor.regular_schedule(0.1))
sim.run(tfinal=30)
data, meta = sim.samples(handle)[0]
v = data[:, 1]
print(v[-1])

assert abs(v[-1]) < 1e-10
