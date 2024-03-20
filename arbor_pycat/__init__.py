import arbor
from typing import Tuple, List, Union, Literal, NamedTuple, Type
import arbor_pycat._core as acm

class IonInfo(NamedTuple):
    name : str
    write_int_concentration: bool = True
    write_ext_concentration: bool = False
    use_diff_concentration: bool = False
    write_rev_potential: bool = False
    read_rev_potential: bool = True
    read_valence: bool = False
    verify_valence: bool = False
    expected_valence: int = 1

class CustomMechanism:
    name: str
    globals: Union[List[Tuple[str, str, float]], Tuple[()]] = ()
    state_vars: Union[List[Tuple[str, str, float]], Tuple[()]] = ()
    parameters: Union[List[Tuple[str, str, float]], Tuple[()]] = ()
    ions: Union[List[IonInfo], Tuple[()]] = ()
    kind: Literal['density', 'point'] = 'density'

    def init_mechanism(self, pp):
        pass

    def advance_state(self, pp):
        pass

    def compute_currents(self, pp):
        pass

    def write_ions(self, pp):
        pass

class PointerPack:
    def __init__(self, pp):
        self.pp = pp
    def _set(self, pp):
        self.pp = pp
        self.dt = pp.dt
        self.width = pp.width
        return self

def register(Mech: Type[CustomMechanism]):
    # Mech = dataclass(frozen=True)(Mech)
    mech = Mech()
    arb_mech = acm.ArbMech();
    def setter(arr, val):
        arr[:] = val
    class SubPointerPack(PointerPack):
        @property
        def node_index(self): return self.pp.node_index
        @property
        def v(self): return self.pp.v
        @v.setter
        def v(self, v): self.pp.v[:] = v
        @property
        def i(self): return self.pp.i
        @i.setter
        def i(self, i): self.pp.i[:] = i
        @property
        def g(self): return self.pp.g
        @g.setter
        def g(self, g): self.pp.g[:] = g

    for name, unit, defaultval in Mech.globals:
        idx = arb_mech.add_global(name, unit, defaultval)
        setattr(SubPointerPack, name, property(lambda self, idx=idx: self.pp.glob(idx)))
    for name, unit, defaultval in Mech.state_vars:
        idx = arb_mech.add_state(name, unit, defaultval)
        f = property(lambda self, idx=idx: self.pp.state(idx))
        f = f.setter(lambda self, val, idx=idx: setter(self.pp.state(idx), val))
        setattr(SubPointerPack, name, f)
    for ioninfo in Mech.ions:
        idx = arb_mech.add_ion(**ioninfo._asdict())
        setattr(SubPointerPack, f'i{ioninfo.name}', property(lambda self, idx=idx: self.pp.ions(idx).current_density).setter(lambda self, val, idx=idx: setter(self.pp.ions(idx).current_density, val)))
        setattr(SubPointerPack, f'c{ioninfo.name}', property(lambda self, idx=idx: self.pp.ions(idx).conductivity).setter(lambda self, val, idx=idx: setter(self.pp.ions(idx).conductivity, val)))
        setattr(SubPointerPack, f'e{ioninfo.name}', property(lambda self, idx=idx: self.pp.ions(idx).reversal_potential).setter(lambda self, val, idx=idx: setter(self.pp.ions(idx).reversal_potential, val)))
        setattr(SubPointerPack, f'{ioninfo.name}i', property(lambda self, idx=idx: self.pp.ions(idx).internal_concentration).setter(lambda self, val, idx=idx: setter(self.pp.ions(idx).internal_concentration, val)))
        setattr(SubPointerPack, f'{ioninfo.name}o', property(lambda self, idx=idx: self.pp.ions(idx).external_concentration).setter(lambda self, val, idx=idx: setter(self.pp.ions(idx).external_concentration, val)))
        setattr(SubPointerPack, f'{ioninfo.name}d', property(lambda self, idx=idx: self.pp.ions(idx).diffusive_concentration).setter(lambda self, val, idx=idx: setter(self.pp.ions(idx).diffusive_concentration, val)))
        setattr(SubPointerPack, f'{ioninfo.name}q', property(lambda self, idx=idx: self.pp.ions(idx).ionic_charge))
        setattr(SubPointerPack, f'index_{ioninfo.name}', property(lambda self, idx=idx: self.pp.ions(idx).index))
    spp = SubPointerPack(None)
    arb_mech.set_init(lambda pp: mech.init_mechanism(spp._set(pp)))
    arb_mech.set_advance_state(lambda pp: mech.advance_state(spp._set(pp)))
    arb_mech.set_compute_currents(lambda pp: mech.compute_currents(spp._set(pp)))
    arb_mech.set_write_ions(lambda pp: mech.write_ions(spp._set(pp)))
    arb_mech.set_name(mech.name)
    acm.register(arb_mech)

def build():
    so_name = acm.get_so_name()
    cat = arbor.load_catalogue(so_name)
    return cat
