#include <limits>
#include <stdexcept>
#include <memory>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <dlfcn.h>

namespace py = pybind11;
using namespace pybind11::literals;

#include <vector>
#include <arbor/mechanism_abi.h>

bool frozen = false;

void frozen_check(bool check = false) {
    if (frozen != check) {
        if (check) throw std::runtime_error("can not modify after catalogue load");
        else throw std::runtime_error("can not call function before load");
    }
}

class ArbIonState {
public:
    ssize_t size;
    arb_ion_state * raw;
    ArbIonState(ssize_t size, arb_ion_state * raw) : size(size), raw(raw) {}
};

template<typename T>
class ArbPPArray {
public:
    ssize_t size;
    T * raw;
    bool ro = false;
    ArbPPArray(size_t size, T * raw, bool ro=false) : size(size), raw(raw), ro(ro) {}
    py::array_t<T> to_numpy() {
        if (!raw) throw std::runtime_error("trying to make a nullpointer into a numpy array");
        return py::array_t<T>(size, raw, py::none());
    }
};

class ArbMech;

class PP {
    arb_mechanism_ppack* pp;
    std::shared_ptr<ArbMech> mech; // just for index check, could remove for performance
public:
    PP(arb_mechanism_ppack* pp, std::shared_ptr<ArbMech> mech) : pp(pp), mech(mech) {}
    ssize_t get_width() { return pp->width; }
    double get_dt() { return pp->dt; }
    py::array_t<arb_index_type> node_index() { return ArbPPArray(get_width(), pp->node_index, true).to_numpy(); }
    arb_value_type glob(int idx) { return pp->globals[idx]; }
    py::array_t<arb_value_type> v(){ return ArbPPArray<arb_value_type>(get_width(), pp->vec_v).to_numpy(); }
    py::array_t<arb_value_type> i(){ return ArbPPArray<arb_value_type>(get_width(), pp->vec_i).to_numpy(); }
    py::array_t<arb_value_type> g(){ return ArbPPArray<arb_value_type>(get_width(), pp->vec_g).to_numpy(); }
    py::array_t<arb_value_type> t_degC(){ return ArbPPArray<arb_value_type>(get_width(), pp->temperature_degC).to_numpy(); }
    py::array_t<arb_value_type> diam_um(){ return ArbPPArray<arb_value_type>(get_width(), pp->diam_um).to_numpy(); }
    py::array_t<arb_value_type> area_um2(){ return ArbPPArray<arb_value_type>(get_width(), pp->area_um2).to_numpy(); }
    py::array_t<arb_value_type> state(size_t idx);
    py::array_t<arb_value_type> param(size_t idx);
    py::array_t<arb_value_type> random(size_t idx);
    ArbIonState ions(size_t idx);
};

class ArbMech {
    std::vector<std::vector<char>> _intern;
    const char * intern(const std::string & s) {
        std::vector<char> copy;
        std::copy( s.begin(), s.end(), std::back_inserter(copy));
        copy.push_back('\0');
        _intern.push_back(copy);
        return _intern.at(_intern.size()-1).data();
    }
public:
    std::string name = "mech";
    arb_mechanism_kind kind = arb_mechanism_kind_density;
    bool is_linear = false;
    bool has_post_events = false;
    std::vector<arb_field_info> globals;
    std::vector<arb_ion_info> ions;
    std::vector<arb_field_info> state_vars;
    std::vector<arb_field_info> parameters;
    std::vector<arb_random_variable_info> random_variables;
    std::function<void(const PP pp)> init_handler;
    std::function<void(const PP pp)> advance_state_handler;
    std::function<void(const PP pp)> compute_currents_handler;
    std::function<void(const PP pp)> write_ions_handler;
    int add_global(const std::string & name, const std::string & unit = "", double default_value=0.) {
        frozen_check();
        arb_field_info afi = { intern(name), intern(unit), default_value, -std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
        globals.push_back(afi);
        return globals.size() - 1;
    }
    int add_state(const std::string & name, const std::string & unit = "", double default_value=0.) {
        frozen_check();
        arb_field_info afi = { intern(name), intern(unit), default_value, -std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
        state_vars.push_back(afi);
        return state_vars.size() - 1;
    }
    int add_parameter(const std::string & name, const std::string & unit = "", double default_value=0.) {
        frozen_check();
        arb_field_info afi = { intern(name), intern(unit), default_value, -std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
        parameters.push_back(afi);
        return parameters.size() - 1;
    }
    int add_random(const std::string & name, ssize_t index) {
        frozen_check();
        // i have no idea what index does
        arb_random_variable_info arvi = { intern(name), (arb_size_type)index };
        random_variables.push_back(arvi);
        return random_variables.size() - 1;
    }
    int add_ion(const std::string & name,
        bool write_int_concentration,
        bool write_ext_concentration,
        bool use_diff_concentration,
        bool write_rev_potential,
        bool read_rev_potential,
        bool read_valence,
        bool verify_valence,
        int  expected_valence
    ) {
        frozen_check();
        arb_ion_info aii = { intern(name),
            write_int_concentration,
            write_ext_concentration,
            use_diff_concentration,
            write_rev_potential,
            read_rev_potential,
            read_valence,
            verify_valence,
            expected_valence };
        ions.push_back(aii);
        return ions.size() - 1;
    }
};

py::array_t<arb_value_type> PP::state(size_t idx) {
    if (!pp->state_vars) throw std::runtime_error("empty state_vars");
    if (idx >= mech->state_vars.size()) throw std::runtime_error("state out of range");
    return ArbPPArray<arb_value_type>(get_width(), pp->state_vars[idx]).to_numpy(); }
py::array_t<arb_value_type> PP::param(size_t idx) {
    if (!pp->parameters) throw std::runtime_error("empty parameters");
    if (idx >= mech->parameters.size()) throw std::runtime_error("param out of range");
    return ArbPPArray<arb_value_type>(get_width(), pp->parameters[idx]).to_numpy(); }
py::array_t<arb_value_type> PP::random(size_t idx) {
    if (!pp->random_numbers) throw std::runtime_error("empty random");
    if (idx >= mech->random_variables.size()) throw std::runtime_error("param out of range");
    // ugly: we discard the const
    return ArbPPArray<arb_value_type>(get_width(), (arb_value_type*)pp->random_numbers[idx], true).to_numpy(); }
ArbIonState PP::ions(size_t idx) {
    if (!pp->ion_states) throw std::runtime_error("empty ion_states");
    if (idx >= mech->ions.size()) throw std::runtime_error("param out of range");
    return ArbIonState(get_width(), &pp->ion_states[idx]); }

std::vector<std::shared_ptr<ArbMech>> mechs;

static void init(arb_mechanism_ppack* pp) {
    // i have absolutely NO idea why, but mechanism_id's are assigned
    // in reverse order??
    int idx = mechs.size() - pp->mechanism_id - 1;
    mechs.at(idx)->init_handler(PP(pp, mechs.at(idx)));
}
static void advance_state(arb_mechanism_ppack* pp) {
    int idx = mechs.size() - pp->mechanism_id - 1;
    mechs.at(idx)->advance_state_handler(PP(pp, mechs.at(idx)));
}
static void compute_currents(arb_mechanism_ppack* pp) {
    int idx = mechs.size() - pp->mechanism_id - 1;
    mechs.at(idx)->compute_currents_handler(PP(pp, mechs.at(idx)));
}
static void write_ions(arb_mechanism_ppack* pp) {
    int idx = mechs.size() - pp->mechanism_id - 1;
    mechs.at(idx)->write_ions_handler(PP(pp, mechs.at(idx)));
}
static void apply_events(arb_mechanism_ppack* pp, arb_deliverable_event_stream* stream_ptr) {
    (void)pp;
    (void)stream_ptr;
}
static void post_event(arb_mechanism_ppack*pp) {
    (void)pp;
}

arb_mechanism_interface * null_interface() { return nullptr; }

arb_mechanism_interface * make_cpu_iface() {
    frozen_check(true);
    static arb_mechanism_interface result;
    result.partition_width = 1;
    result.backend = arb_backend_kind_cpu;
    result.alignment = 8;
    result.init_mechanism   = init;
    result.compute_currents = compute_currents;
    result.apply_events     = apply_events;
    result.advance_state    = advance_state;
    result.write_ions       = write_ions;
    result.post_event       = post_event;
    return &result;
}

arb_mechanism_type make_input_type() {
    frozen_check(true);
    static size_t call_counter = 0;
    // this is ugly, but we know this function is called in a loop
    // so in that way we find out which mechanism we are generating for
    if (call_counter >= mechs.size()) throw std::runtime_error("called > size(mechs) times; arbor internals changed?");
    auto & mech = mechs.at(call_counter);
    call_counter += 1;
    arb_mechanism_type result;
    result.abi_version = ARB_MECH_ABI_VERSION;
    result.fingerprint = "<placeholder>";
    result.name = mech->name.c_str();
    result.kind = mech->kind;
    result.is_linear = mech->is_linear;
    result.has_post_events = mech->has_post_events;
    result.globals = mech->globals.data();
    result.n_globals = mech->globals.size();
    result.ions = mech->ions.data();
    result.n_ions = mech->ions.size();
    result.state_vars = mech->state_vars.data();
    result.n_state_vars = mech->state_vars.size();
    result.parameters = mech->parameters.data();
    result.n_parameters = mech->parameters.size();
    result.random_variables = mech->random_variables.data();
    result.n_random_variables = mech->random_variables.size();
    return result;
}

extern "C" [[gnu::visibility("default")]] const void* get_catalogue(int* n) {
    /* arbor entry point */
    static arb_mechanism mechanism_template;
    static std::vector<arb_mechanism> mechanisms;
    mechanism_template.type = make_input_type,
    mechanism_template.i_cpu = make_cpu_iface,
    mechanism_template.i_gpu = null_interface;
    for (size_t i = 0; i < mechs.size(); i++) {
        mechanisms.push_back(mechanism_template);
    }
    *n = mechs.size();
    frozen = true;
    return (void*)mechanisms.data();
}

const char * get_so_name() {
    Dl_info DlInfo;
    if(!dladdr((void*)get_so_name, &DlInfo)) {
        return "";
    }
    return DlInfo.dli_fname;
}

PYBIND11_MODULE(_core, m) {
    /* pybind entry point */
    m.doc() = "Custom Arbor Mod";
    m.def("get_so_name", []() {
        const char * so_name = get_so_name();
        return std::string(so_name);
        });

    m.def("register", [](std::shared_ptr<ArbMech> & mech) {
        frozen_check();
        mechs.push_back(mech);
    });
    py::class_<ArbMech, std::shared_ptr<ArbMech>>(m, "ArbMech")
        .def(py::init<>())
        .def("set_name", [](std::shared_ptr<ArbMech> & mech, const std::string & name) {
            mech->name = name;
        })
        .def("add_global", [](std::shared_ptr<ArbMech> & mech, const std::string & name, const std::string & unit, double defaultval) {
            return mech->add_global(name, unit, defaultval);
        })
        .def("add_state", [](std::shared_ptr<ArbMech> & mech, const std::string & name, const std::string & unit, double defaultval) {
            return mech->add_state(name, unit, defaultval);
        })
        .def("add_parameter", [](std::shared_ptr<ArbMech> & mech, const std::string & name, const std::string & unit, double defaultval) {
            return mech->add_parameter(name, unit, defaultval);
        })
        .def("add_random", [](std::shared_ptr<ArbMech> & mech, const std::string & name, size_t index) {
            return mech->add_random(name, index);
        })
        .def("add_ion",
                ([](
                std::shared_ptr<ArbMech> & mech, 
                const std::string & name,
                bool write_int_concentration,
                bool write_ext_concentration,
                bool use_diff_concentration,
                bool write_rev_potential,
                bool read_rev_potential,
                bool read_valence,
                bool verify_valence,
                int  expected_valence
                    ) {
            return mech->add_ion(
                name,
                write_int_concentration,
                write_ext_concentration,
                use_diff_concentration,
                write_rev_potential,
                read_rev_potential,
                read_valence,
                verify_valence,
                expected_valence
            ); }),
                py::arg("name"),
                py::arg("write_int_concentration") = true,
                py::arg("write_ext_concentration") = false,
                py::arg("use_diff_concentration") = false,
                py::arg("write_rev_potential") = false,
                py::arg("read_rev_potential") = true,
                py::arg("read_valence") = false,
                py::arg("verify_valence") = false,
                py::arg("expected_valence") = 1
                )
        .def("set_kind_point", [](std::shared_ptr<ArbMech> & mech) {
            frozen_check();
            mech->kind = arb_mechanism_kind_density;
        })
        .def("set_kind_point", [](std::shared_ptr<ArbMech> & mech) {
            frozen_check();
            mech->kind = arb_mechanism_kind_density;
        })
        .def("set_init", [](std::shared_ptr<ArbMech> & mech, std::function<void(const PP pp)> init_handler) {
            mech->init_handler = init_handler;
        })
        .def("set_advance_state", [](std::shared_ptr<ArbMech> & mech, std::function<void(const PP pp)> advance_state_handler) {
            mech->advance_state_handler = advance_state_handler;
        })
        .def("set_compute_currents", [](std::shared_ptr<ArbMech> & mech, std::function<void(const PP pp)> compute_currents_handler) {
            mech->compute_currents_handler = compute_currents_handler;
        })
        .def("set_write_ions", [](std::shared_ptr<ArbMech> & mech, std::function<void(const PP pp)> write_ions_handler) {
            mech->write_ions_handler = write_ions_handler;
        });
    m.add_object("_cleanup", py::capsule([]() {
        /* prevent segfault */
        for (auto & mech : mechs) {
            mech->init_handler = {};
            mech->init_handler = {};
            mech->advance_state_handler = {};
            mech->compute_currents_handler = {};
            mech->write_ions_handler = {};
        }
    }));
    py::class_<PP>(m, "PP")
        .def_property_readonly("width", &PP::get_width)
        .def_property_readonly("dt", &PP::get_dt)
        .def_property_readonly("node_index", &PP::node_index)
        .def("state", &PP::state)
        .def("glob", &PP::glob)
        .def("param", &PP::param)
        .def("random", &PP::random)
        .def_property_readonly("v", &PP::v)
        .def_property_readonly("i", &PP::i)
        .def_property_readonly("g", &PP::g)
        .def_property_readonly("t_degC", &PP::t_degC)
        .def_property_readonly("diam_um", &PP::diam_um)
        .def_property_readonly("area_um2", &PP::area_um2)
        .def("ions", &PP::ions)
        ;
    py::class_<ArbPPArray<arb_index_type>>(m, "ArborPPIndexArray", py::buffer_protocol())
        .def_buffer([](ArbPPArray<arb_index_type> & p) {
            return py::buffer_info(p.raw, p.size, p.ro);
        });
    py::class_<ArbPPArray<arb_value_type>>(m, "ArborPPDoubleArray", py::buffer_protocol())
        .def_buffer([](ArbPPArray<arb_value_type> & p) {
            return py::buffer_info(p.raw, p.size, p.ro);
        });
    py::class_<ArbIonState>(m, "ArbIonState")
        .def_property_readonly("current_density", [](ArbIonState & s) { return ArbPPArray<arb_value_type>(s.size, s.raw->current_density).to_numpy(); })
        .def_property_readonly("conductivity", [](ArbIonState & s) { return ArbPPArray<arb_value_type>(s.size, s.raw->conductivity).to_numpy(); })
        .def_property_readonly("reversal_potential", [](ArbIonState & s) { return ArbPPArray<arb_value_type>(s.size, s.raw->reversal_potential).to_numpy(); })
        .def_property_readonly("internal_concentration", [](ArbIonState & s) { return ArbPPArray<arb_value_type>(s.size, s.raw->internal_concentration).to_numpy(); })
        .def_property_readonly("external_concentration", [](ArbIonState & s) { return ArbPPArray<arb_value_type>(s.size, s.raw->external_concentration).to_numpy(); })
        .def_property_readonly("diffusive_concentration", [](ArbIonState & s) { return ArbPPArray<arb_value_type>(s.size, s.raw->diffusive_concentration).to_numpy(); })
        .def_property_readonly("ionic_charge", [](ArbIonState & s) { return ArbPPArray<arb_value_type>(s.size, s.raw->ionic_charge).to_numpy(); })
        .def_property_readonly("index", [](ArbIonState & s) { return ArbPPArray<arb_index_type>(s.size, s.raw->index).to_numpy(); })
        ;
#ifdef VERSION_INFO
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
