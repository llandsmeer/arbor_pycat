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

## Installation

```
pip install git+https://github.com/llandsmeer/arbor_pycat.git#egg=arbor_pycat
```

## Known bugs

The default accessors (`Xi`, `Xo`, `iX` etc) for ions ignore the index attribute

## Example

```python
import arbor
import arbor_pycat

@arbor_pycat.register
class Passive(arbor_pycat.CustomMechanism):
    name = 'passive'
    def init_mechanism(self, pp):
        print(dir(pp))
    def compute_currents(self, pp):
        pp.i = pp.v * 1e-2

@arbor_pycat.register
class ExampleMech(arbor_pycat.CustomMechanism):
    name = 'example'
    state_vars = [('x', 'mV', 1.),
                  ('y', 'mV', 0.)]
    ions = [arbor_pycat.IonInfo('ca', expected_valence=2, verify_valence=True)]

    def init_mechanism(self, pp):
        print(dir(pp))
        pp.v = 10

    def advance_state(self, pp):
        pp.x +=  pp.y * pp.dt
        pp.y += -pp.x * pp.dt

    def compute_currents(self, pp):
        pp.i = pp.v + pp.x

    def write_ions(self, pp):
        pp.eka = +80
        pp.cai = -pp.v

cat = arbor_pycat.build()
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
        pp.m = alpha_m(pp.v) / (alpha_m(pp.v) + beta_m(pp.v))
        pp.h = alpha_h(pp.v) / (alpha_h(pp.v) + beta_h(pp.v))
        pp.n = alpha_n(pp.v) / (alpha_n(pp.v) + beta_n(pp.v))

    def advance_state(self, pp):
        pp.t += pp.dt
        pp.m += pp.dt * (alpha_m(pp.v)*(1-pp.m) - beta_m(pp.v)*pp.m)
        pp.h += pp.dt * (alpha_h(pp.v)*(1-pp.h) - beta_h(pp.v)*pp.h)
        pp.n += pp.dt * (alpha_n(pp.v)*(1-pp.n) - beta_n(pp.v)*pp.n)

    def compute_currents(self, pp):
        ina = pp.gna*pp.m**3*pp.h*(pp.v - pp.ena)
        ik = pp.gk*pp.n**4*(pp.v - pp.ek)
        il = pp.gl*(pp.v - pp.el)
        iapp = 0.05 if pp.t[0] % 100 > 95 else 0
        pp.i = ina + ik + il - iapp
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
