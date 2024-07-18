# (Experimental) custom mechanisms in python for Arbor

![CI](https://github.com/llandsmeer/arbor_pycat/actions/workflows/python-package.yml/badge.svg)

Allows one to define mechanisms in python instead of NMODL or C++

VERY experimental and full of footguns. Likely to break with arbor updates.
Makes assumptions on the pointer pack that definitely are not true, and allows
you to edit currents, voltages and ions states in ways that might break the arbor solver.

Known to work for `arbor==0.9.0`

The overhead for calling back to python is quite large at the moment:
For a single CV cell, initial tests with the HH model suggests that the call overhead is about 600x over the builtin HH NMODL implementation.
For a 1024 CV cell and JAX.jit compilation, this is reduced to ~1.7 times slower execution.

Make sure you use the right indexing method! Else you will access memory you should not access...

For example:

 - `pp.v[pp.node_index]`, `pp.i[pp.node_index]` (`v`, `g`, `i` via node index)
 - `pp.state` (state as is)
 - `pp.ica[pp.index_ca]`,`pp.ek[pp.index_k]` (ions via ion index)

## Installation

```
pip install git+https://github.com/llandsmeer/arbor_pycat.git#egg=arbor_pycat
```

## Example

```python
import arbor
import arbor_pycat

@arbor_pycat.register
class Passive(arbor_pycat.CustomMechanism):
    name = 'passive'
    def init_mechanism(self, pp):
        pass
    def compute_currents(self, pp):
        pp.i[pp.node_index] = pp.v[pp.node_index] * 1e-2

@arbor_pycat.register
class ExampleMech(arbor_pycat.CustomMechanism):
    name = 'example'
    state_vars = [('x', 'mV', 1.),
                  ('y', 'mV', 0.)]
    ions = [arbor_pycat.IonInfo('ca', expected_valence=2, verify_valence=True)]

    def init_mechanism(self, pp):
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

# [...]

decor = (
    arbor.decor()
    .paint('"soma"', arbor.density("example"))
    .paint('"dend"', arbor.density("passive"))
)

props = arbor.neuron_cable_properties()
props.catalogue.extend(cat, '')
```

## Hodgkin-Huxley example

```python
def exprelr(x): return np.where(np.isclose(x, 0), 1., x / np.expm1(x))
def alpha_m(V): return exprelr(-0.1*V - 4.0)
def alpha_h(V): return 0.07*np.exp(-0.05*V - 3.25)
def alpha_n(V): return 0.1*exprelr(-0.1*V - 5.5)
def beta_m(V):  return 4.0*np.exp(-(V + 65.0)/18.0)
def beta_h(V):  return 1.0/(np.exp(-0.1*V - 3.5) + 1.0)
def beta_n(V):  return 0.125*np.exp(-0.0125*V - 0.8125)

@arbor_pycat.register
class ExampleMech(arbor_pycat.CustomMechanism):
    name = 'custom_hh'
    kind = 'density'
    state_vars = [('m', '', 0), ('h', '', 0), ('n', '', 0), ('t', '', 0)]
    parameters = [('gna', '',   0.120),
                  ('gk', '',    0.036),
                  ('gl', '',    0.0003),
                  ('ena', '',  55),
                  ('ek', '',  -77),
                  ('el', '',  -65)]

    def init_mechanism(self, pp):
        pp.m = alpha_m(pp.v[pp.node_index]) / (alpha_m(pp.v[pp.node_index]) + beta_m(pp.v[pp.node_index]))
        pp.h = alpha_h(pp.v[pp.node_index]) / (alpha_h(pp.v[pp.node_index]) + beta_h(pp.v[pp.node_index]))
        pp.n = alpha_n(pp.v[pp.node_index]) / (alpha_n(pp.v[pp.node_index]) + beta_n(pp.v[pp.node_index]))

    def advance_state(self, pp):
        pp.t += pp.dt
        pp.m += pp.dt * (alpha_m(pp.v)*(1-pp.m) - beta_m(pp.v)*pp.m)
        pp.h += pp.dt * (alpha_h(pp.v)*(1-pp.h) - beta_h(pp.v)*pp.h)
        pp.n += pp.dt * (alpha_n(pp.v)*(1-pp.n) - beta_n(pp.v)*pp.n)

    def compute_currents(self, pp):
        ina = pp.gna*pp.m**3*pp.h*(pp.v[pp.node_index] - pp.ena)
        ik = pp.gk*pp.n**4*(pp.v[pp.node_index] - pp.ek)
        il = pp.gl*(pp.v[pp.node_index] - pp.el)
        iapp = 0.05 if pp.t[0] % 100 > 95 else 0
        pp.i[pp.node_index] = ina + ik + il - iapp
```

## Debugging segfaults

```
pip install .
mkdir build
cd build
cmake -DPython_EXECUTABLE:FILEPATH=$(which python3) -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-I ~/.local/lib/python3.10/site-packages/arbor/include" ..
mv _core.cpython-310-x86_64-linux-gnu.so ~/.local/lib/python3.10/site-packages/arbor_pycat/_core.cpython-310-x86_64-linux-gnu.so

# [ ... ]

gdb --arg python3 api.py
```

# Inferior Olive (de Gruijl model) example

```
import jax
import jax.numpy as jnp
import arbor_pycat._core as acm

# Channel conductance parameters
g_CaL           = 0.045      # Calcium T - (CaV 3.1) (0.7)
g_h             = 0.015      # H current (HCN) (0.4996)
g_K_Ca          = 0.220      # Potassium  (KCa v1.1 - BK) (35)
g_ld            = 1.3e-5     # Leak dendrite (0.016)
g_la            = 1.3e-5     # Leak axon (0.016)
g_ls            = 1.3e-5     # Leak soma (0.016)
g_Na_s          = 0.030      # Sodium  - (Na v1.6 )
g_Kdr_s         = 0.030      # Potassium - (K v4.3)
g_K_s           = 0.015      # Potassium - (K v3.4)
g_CaH           = 0.010      # High-threshold calcium -- Ca V2.1
g_Na_a          = 0.200      # Sodium
g_K_a           = 0.200      # Potassium (20)

# Reversal potential parameters
V_Na            =  55.0      # Sodium
V_K             = -75.0      # Potassium
V_Ca            = 120.0      # Low-threshold calcium channel
V_h             = -43.0      # H current
V_l             =  10.0      # Leak

def single_cell():
    import arbor
    import matplotlib.pyplot as plt
    nml = arbor.neuroml('./C51A.cell.nml').cell_morphology('C51A')
    label = arbor.label_dict()
    label.append(nml.segments())
    label.append(nml.named_segments())
    label.append(nml.groups())
    decor = (
        arbor.decor()
        .paint('"soma_group"', arbor.density('io_soma_fwd'))
        .paint('"dendrite_group"', arbor.density('io_dend_fwd'))
        .paint('"axon_group"', arbor.density('io_axon_fwd'))
        .set_property(cm=0.01, rL=100, Vm=-60)
    )
    cell = arbor.cable_cell(nml.morphology, decor, label)
    m = arbor.single_cell_model(cell)
    m.properties.catalogue.extend(build(), '')
    m.probe('voltage', where='(root)', frequency=1) 
    m.run(2000, 0.025*4)
    t = m.traces[0].time
    v = m.traces[0].value
    plt.plot(t, v)
    plt.show()

class Soma:
    state = 'k', 'l', 'h', 'n', 'x'
    @staticmethod
    def init(v):
        k           = 1 / (1 + jnp.exp(-(v + 61)/4.2))
        l           = 1 / (1 + jnp.exp( (v + 85)/8.5))
        h           = 1 / (1 + jnp.exp( (v + 70)/5.8))
        n           = 1 / ( 1 + jnp.exp(-(v +  3)/10))
        alpha_x     = 0.13 * (v + 25) / (1 - jnp.exp(-(v + 25)/10))
        beta_x      = 1.69 * jnp.exp(-(v + 35)/80)
        x           = alpha_x / (alpha_x + beta_x)
        return jnp.stack([k, l, h, n, x])
    @staticmethod
    def compute_current(v, state):
        k, l, h, n, x = state
        I_leak = g_ls * (v - V_l)
        Ical   = g_CaL * k * k * k * l * (v - V_Ca)
        m_inf  = 1 / (1 + jnp.exp(-(v + 30)/5.5))
        Ina    = g_Na_s * m_inf**3 * h * (v - V_Na)
        Ikdr   = g_Kdr_s * n**4 * (v - V_K)
        Ik     = g_K_s * x**4 * (v - V_K)
        return I_leak + Ik + Ikdr + Ina + Ical
    @staticmethod
    def state_gradient(v, state):
        k, l, h, n, x = state
        k_inf       = 1 / (1 + jnp.exp(-(v + 61)/4.2))
        l_inf       = 1 / (1 + jnp.exp( (v + 85)/8.5))
        tau_l       = (20 * jnp.exp((v + 160)/30) / (1 + jnp.exp((v + 84) / 7.3))) + 35
        dk_dt       = k_inf - k
        dl_dt       = (l_inf - l) / tau_l
        h_inf       = 1 / (1 + jnp.exp( (v + 70)/5.8))
        tau_h       = 3 * jnp.exp(-(v + 40)/33)
        dh_dt       = (h_inf - h) / tau_h
        n_inf       = 1 / ( 1 + jnp.exp(-(v +  3)/10))
        tau_n       = 5 + (47 * jnp.exp( (v + 50)/900))
        dn_dt       = (n_inf - n) / tau_n
        alpha_x     = 0.13 * (v + 25) / (1 - jnp.exp(-(v + 25)/10))
        beta_x      = 1.69 * jnp.exp(-(v + 35)/80)
        tau_x_inv   = alpha_x + beta_x
        x_inf       = alpha_x / tau_x_inv
        dx_dt       = (x_inf - x) * tau_x_inv
        return jnp.stack([dk_dt, dl_dt, dh_dt, dn_dt, dx_dt])

class Dend:
    state = 'caconc', 'r', 's', 'q'
    @staticmethod
    def init(v):
        caconc      =  jnp.full_like(v, 3.715)
        alpha_r     =  1.7 / (1 + jnp.exp(-(v - 5)/13.9))
        beta_r      =  0.02*(v + 8.5) / (jnp.exp((v + 8.5)/5) - 1.0)
        r           =  alpha_r / (alpha_r + beta_r)
        alpha_s     =  jnp.where(0.00002 * caconc < 0.01, 0.00002 * caconc, 0.01)
        s           =  alpha_s / (alpha_s + 0.015)
        q           =  1 / (1 + jnp.exp((v + 80)/4))
        return jnp.stack([caconc, r, s, q])
    @staticmethod
    def compute_current(v, state):
        _, r, s, q  = state
        I_leak      =  g_ld * (v - V_l)
        Icah        =  g_CaH * r * r * (v - V_Ca) * 0
        Ikca        =  g_K_Ca * s * (v - V_K)
        Ih          =  g_h * q * (v - V_h)
        return I_leak + Icah + Ikca + Ih
    @staticmethod
    def state_gradient(v, state):
        caconc, r, s, q = state
        Icah        =  g_CaH * r * r * (v - V_Ca)
        alpha_r     =  1.7 / (1 + jnp.exp(-(v - 5)/13.9))
        beta_r      =  0.02*(v + 8.5) / (jnp.exp((v + 8.5)/5) - 1.0)
        tau_r_inv5  =  (alpha_r + beta_r)
        r_inf       =  alpha_r / tau_r_inv5
        dr_dt       =  (r_inf - r) * tau_r_inv5 * 0.2
        alpha_s     =  jnp.where(
                0.00002 * caconc < 0.01,
                0.00002 * caconc,
                0.01)
        tau_s_inv   =  alpha_s + 0.015
        s_inf       =  alpha_s / tau_s_inv
        ds_dt       =  (s_inf - s) * tau_s_inv
        q_inf       =  1 / (1 + jnp.exp((v + 80)/4))
        tau_q_inv   =  jnp.exp(-0.086*v - 14.6) + jnp.exp(0.070*v - 1.87)
        dq_dt       =  (q_inf - q) * tau_q_inv
        dCa_dt      =  -3 * Icah - 0.075 * caconc
        return jnp.stack([dCa_dt, dr_dt, ds_dt, dq_dt])

class Axon:
    state = 'h', 'x'
    @staticmethod
    def init(v):
        h     =  1 / (1 + jnp.exp( (v+60)/5.8))
        alpha_x   =  0.13*(v + 25) / (1 - jnp.exp(-(v + 25)/10))
        beta_x    =  1.69 * jnp.exp(-(v + 35)/80)
        tau_x_inv =  alpha_x + beta_x
        x     =  alpha_x / tau_x_inv
        return jnp.stack([h, x])
    @staticmethod
    def compute_current(v, state):
        h, x = state
        m_inf     =  1 / (1 + jnp.exp(-(v+30)/5.5))
        I_leak    =  g_la * (v - V_l)
        Ina       =  g_Na_a * m_inf**3 * h * (v - V_Na)
        Ik        =  g_K_a * x**4 * (v - V_K)
        return I_leak + Ina + Ik
    @staticmethod
    def state_gradient(v, state):
        h, x = state
        h_inf     =  1 / (1 + jnp.exp( (v+60)/5.8))
        tau_h     =  1.5 * jnp.exp(-(v+40)/33)
        dh_dt     =  (h_inf - h) / tau_h
        alpha_x   =  0.13*(v + 25) / (1 - jnp.exp(-(v + 25)/10))
        beta_x    =  1.69 * jnp.exp(-(v + 35)/80)
        tau_x_inv =  alpha_x + beta_x
        x_inf     =  alpha_x / tau_x_inv
        dx_dt     =  (x_inf - x) * tau_x_inv
        return jnp.stack([dh_dt, dx_dt])

def forward(decl, mech_name):
    n = len(decl.state)
    arb_mech = acm.ArbMech()
    for i, name in enumerate(decl.state):
        assert i == arb_mech.add_state(name, '', 0.)
    def get_state(pp): return jnp.stack([pp.state(i) for i in range(n)])
    def set_state(pp, val):
        for i in range(n): pp.state(i)[:] = val[i]
        return jnp.stack([pp.state(i) for i in range(n)])
    def init(pp):
        v = pp.v[pp.node_index]
        val = decl.init(v)
        set_state(pp, val)
    jit_state_gradient = jax.jit(decl.state_gradient)
    def advance_state(pp):
        v = pp.v[pp.node_index]
        x = get_state(pp)
        x = x + pp.dt * jit_state_gradient(v, x)
        set_state(pp, x)
    jit_compute_current = jax.jit(decl.compute_current)
    def compute_currents(pp):
        v = pp.v[pp.node_index]
        x = get_state(pp)
        i = jit_compute_current(v, x)
        pp.i[pp.node_index] = i
    arb_mech.set_init(init)
    arb_mech.set_advance_state(advance_state)
    arb_mech.set_compute_currents(compute_currents)
    arb_mech.set_write_ions(lambda _: None)
    arb_mech.set_name(mech_name)
    acm.register(arb_mech)

def register():
    forward(Soma, 'io_soma_fwd')
    forward(Dend, 'io_dend_fwd')
    forward(Axon, 'io_axon_fwd')


def build():
    import arbor
    so_name = acm.get_so_name()
    cat = arbor.load_catalogue(so_name)
    return cat


register()
single_cell()
```
