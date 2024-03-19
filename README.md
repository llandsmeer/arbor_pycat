# (Experimental) custom mechanisms in python for Arbor

Allows one to define mechanisms in python instead of NMODL or C++

VERY experimental and full of footguns

```python
import arbor
import numpy as np
import arbor_custom_mod._core as acm

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
```
