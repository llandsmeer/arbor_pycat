# (Experimental) custom mechanisms in python for Arbor

Allows one to define mechanisms in python instead of NMODL or C++

VERY experimental and full of footguns. Likely to break with arbor updates.
Makes assumptions on the pointer pack that definitely are not true.

Known to work for `arbor==0.9.0`

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
