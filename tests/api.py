import arbor
import arbor_pycat

@arbor_pycat.register
class Passive(arbor_pycat.CustomMechanism):
    name = 'passive'
    def init_mechanism(self, pp):
        print('passive', pp.node_index)
    def compute_currents(self, pp):
        pp.i[pp.node_index] = pp.v[pp.node_index] * 1e-2

@arbor_pycat.register
class ExampleMech(arbor_pycat.CustomMechanism):
    name = 'example'
    state_vars = [('x', 'mV', 1.),
                  ('y', 'mV', 0.)]
    ions = [arbor_pycat.IonInfo('ca', expected_valence=2, verify_valence=True)]

    def init_mechanism(self, pp):
        print('example', pp.node_index)
        pp.v[pp.node_index] = 10

    def advance_state(self, pp):
        dx =  pp.y * pp.dt
        dy = -pp.x * pp.dt
        pp.x += dx
        pp.y += dy
        print(pp.x)

    def compute_currents(self, pp):
        pp.i[pp.node_index] = (pp.v[pp.node_index] + pp.x) * 1e-1

    def write_ions(self, pp):
        pp.eca[pp.index_ca] = +80
        pp.cai[pp.index_ca] = -pp.v[pp.node_index]

cat = arbor_pycat.build()

# import matplotlib.pyplot as plt

tree = arbor.segment_tree()
tree.append(arbor.mnpos, arbor.mpoint(-3, 0, 0, 3), arbor.mpoint(3, 0, 0, 3), tag=1)
tree.append(arbor.mnpos, arbor.mpoint(-3, 0, 0, 3), arbor.mpoint(3, 0, 0, 3), tag=2)
tree.append(arbor.mnpos, arbor.mpoint(-3, 0, 0, 3), arbor.mpoint(3, 0, 0, 3), tag=2)

# (2) Define the soma and its midpoint
labels = arbor.label_dict({"soma": "(tag 1)", "dend": "(tag 2)", "midpoint": "(location 0 0.5)"})

# (3) Create cell and set properties
decor = (
    arbor.decor()
    .set_property(Vm=-40, rL=1e5)
    .paint('"soma"', arbor.density("example"))
    .paint('"dend"', arbor.density("passive"))
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
    def probes(self, _): return [
            arbor.cable_probe_membrane_voltage('(root)'),
            arbor.cable_probe_ion_int_concentration('(root)', 'ca')
            ]
    def global_properties(self, kind): return self.the_props
recipe = single_recipe()
sim = arbor.simulation(recipe)
handle = sim.sample((0, 0), arbor.regular_schedule(0.1))
ca_handle = sim.sample((0, 1), arbor.regular_schedule(0.1))
sim.run(tfinal=30)
cai = sim.samples(ca_handle)[0][0][:,1]
print(cai)
data, meta = sim.samples(handle)[0]
v = data[:, 1]
#import matplotlib.pyplot as plt
#plt.plot(v)
#plt.show()
print(v[-1])
# plt.plot(v)
# plt.plot(cai)
# plt.show()


assert abs(v[-1]) < 2
assert abs(cai[-1]) < 2
