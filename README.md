# (Experimental) custom mechanisms in python for Arbor

Allows one to define mechanisms in python instead of NMODL or C++

VERY experimental and full of footguns. Likely to break with arbor updates.
Makes assumptions on the pointer pack that definitely are not true.
Only one mechanism supported for now (although that should be an easy fix).

## Installation

```
pip install git+https://github.com/llandsmeer/arbor_custom_mod.git#egg=arbor_custom_mod
```

## Example

```python
import arbor
from arbor_custom_mod import IonInfo, CustomMechanism, register

class ExampleMech(CustomMechanism):
    name = 'arbor_custom_mod'
    state_vars = [('x', 'mV', 1),
                  ('y', 'mV', 0)]
    ions = [IonInfo('ca', expected_valence=2, verify_valence=True)]

    def init_mechanism(self, pp):
        pp.v = 10

    def advance_state(self, pp):
        pp.x +=  pp.y * pp.dt
        pp.y += -pp.x * pp.dt

    def compute_currents(self, pp):
        pp.i = pp.v + pp.x

    def write_ions(self, pp):
        pp.eka = +80
        pp.cai = -pp.v

cat = register(ExampleMech)
```
