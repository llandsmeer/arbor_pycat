#include <limits>
#include <stdexcept>
#include <memory>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <unistd.h>
#include <dlfcn.h>

#define ERROR(x) do { \
    char buf[] = "arbor_pycat:" __FILE__ ":" x; \
    write(STDOUT_FILENO, buf, sizeof(buf)); \
    throw std::runtime_error(x); \
} while (0)

/* From arbor documentation:
 * 
    typedef struct {
        // Global data
        arb_index_type width;                           // Number of CVs of this mechanism, size of arrays
        arb_index_type n_detectors;                     // Number of spike detectors
        arb_index_type* vec_ci;                         // [Array] Map CV to cell
        arb_index_type* vec_di;                         // [Array] Map
        const arb_value_type* vec_t;                    // [Array] time value
        arb_value_type* vec_dt;                         // [Array] time step
        arb_value_type* vec_v;                          // [Array] potential
        arb_value_type* vec_i;                          // [Array] current
        arb_value_type* vec_g;                          // [Array] conductance
        arb_value_type* temperature_degC;               // [Array] Temperature in celsius
        arb_value_type* diam_um;                        // [Array] CV diameter
        arb_value_type* time_since_spike;               // Times since last spike; one entry per cell and detector.
        arb_index_type* node_index;                     // Indices of CVs covered by this mechanism, size is width
        arb_index_type* multiplicity;                   // [Unused]
        arb_value_type* weight;                         // [Array] Weight
        arb_size_type mechanism_id;                     // Unique ID for this mechanism on this cell group
        arb_deliverable_event_stream events;            // Events during the last period
        arb_constraint_partition     index_constraints; // Index restrictions, not initialised for all backends.
        // User data
        arb_value_type** parameters;                    // [Array] setable parameters
        arb_value_type** state_vars;                    // [Array] integrable state
        arb_value_type*  globals;                       // global constant state
        arb_ion_state*   ion_states;                    // [Array] views into shared state
    } arb_mechanism_ppack;

Members tagged as ``[Array]`` represent one value per CV. To access the values
belonging to your mechanism, a level of indirection via ``node_index`` is
needed.


simplified load_catalogue logic
ARB_ARBOR_API const mechanism_catalogue load_catalogue(const std::filesystem::path& fn) {
    typedef void* global_catalogue_t(int*);
    global_catalogue_t* get_catalogue = nullptr;
    get_catalogue = util::dl_get_symbol<global_catalogue_t*>(fn, "get_catalogue");
    int count = -1;
    auto mechs = (arb_mechanism*)get_catalogue(&count);
    mechanism_catalogue result;
    for(int ix = 0; ix < count; ++ix) {
        auto type = mechs[ix].type();
        auto name = std::string{type.name};
        auto icpu = mechs[ix].i_cpu();
        auto igpu = mechs[ix].i_gpu();
        result.add(name, type);
        result.register_implementation(name, std::make_unique<mechanism>(type, *icpu));
    }
    return result;
}

*/

namespace py = pybind11;
using namespace pybind11::literals;

#include <vector>
#include <arbor/mechanism_abi.h>

bool frozen = false;

void frozen_check(bool check = false) {
    if (frozen != check) {
        if (check) ERROR("can not modify after catalogue load");
        else ERROR("can not call function before load");
    }
}

class ArbIonState {
public:
    ssize_t size_of_index_array;
    arb_ion_state * raw;
    int size_of_data_arrays;
    ArbIonState(ssize_t size, arb_ion_state * raw) : size_of_index_array(size), raw(raw) {
        int maxidx = 0;
        for (ssize_t i = 0; i < size; i++) {
            if (raw->index[i] > maxidx) {
                maxidx = raw->index[i];
            }
        }
        size_of_data_arrays = maxidx + 1;
    }
};

template<typename T>
class ArbPPArray {
public:
    ssize_t size;
    T * raw;
    bool ro = false;
    ArbPPArray(size_t size, T * raw, bool ro=false) : size(size), raw(raw), ro(ro) {}
    py::array_t<T> to_numpy() {
        if (!raw) ERROR("trying to make a nullpointer into a numpy array");
        return py::array_t<T>(size, raw, py::none());
    }
};

class ArbMech;

class PP {
    arb_mechanism_ppack* pp;
    std::shared_ptr<ArbMech> mech; // just for index check, could remove for performance
    int max_node_index = 0; // node_index has get_width size, but max index into v could be higher
public:
    PP(arb_mechanism_ppack* pp, std::shared_ptr<ArbMech> mech) : pp(pp), mech(mech) {
        // slow, inefficient etc but still O(n). should precalculate
        int maxidx = 0;
        for (size_t i = 0; i < pp->width; i++) {
            if (pp->node_index[i] > maxidx) {
                maxidx = pp->node_index[i];
            }
        }
        max_node_index = maxidx;
    }
    ssize_t get_width() { return pp->width; } //  _pp_var_width
    ssize_t get_nwidth() { return max_node_index + 1; }
    double get_dt() { return pp->dt; } // _pp_var_dt
    py::array_t<arb_index_type> node_index() { return ArbPPArray(get_width(), pp->node_index, true).to_numpy(); }
    arb_value_type glob(int idx) { return pp->globals[idx]; }
    py::array_t<arb_value_type> v(){ return ArbPPArray<arb_value_type>(get_nwidth(), pp->vec_v).to_numpy(); }
    py::array_t<arb_value_type> i(){ return ArbPPArray<arb_value_type>(get_nwidth(), pp->vec_i).to_numpy(); }
    py::array_t<arb_value_type> g(){ return ArbPPArray<arb_value_type>(get_nwidth(), pp->vec_g).to_numpy(); }
    py::array_t<arb_value_type> t_degC(){ return ArbPPArray<arb_value_type>(max_node_index+1, pp->temperature_degC).to_numpy(); }
    py::array_t<arb_value_type> diam_um(){ return ArbPPArray<arb_value_type>(max_node_index+1, pp->diam_um).to_numpy(); }
    py::array_t<arb_value_type> area_um2(){ return ArbPPArray<arb_value_type>(max_node_index+1, pp->area_um2).to_numpy(); }
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
    ArbMech() {
        add_global("_arbor_pycat_mech_idx", "1", 123456789.); // to be updated on register
    }
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
    if (!pp->state_vars) ERROR("empty state_vars");
    if (idx >= mech->state_vars.size()) ERROR("state out of range");
    return ArbPPArray<arb_value_type>(get_width(), pp->state_vars[idx]).to_numpy(); }
py::array_t<arb_value_type> PP::param(size_t idx) {
    if (!pp->parameters) ERROR("empty parameters");
    if (idx >= mech->parameters.size()) ERROR("param out of range");
    return ArbPPArray<arb_value_type>(get_width(), pp->parameters[idx]).to_numpy(); }
py::array_t<arb_value_type> PP::random(size_t idx) {
    if (!pp->random_numbers) ERROR("empty random");
    if (idx >= mech->random_variables.size()) ERROR("param out of range");
    // ugly: we discard the const
    return ArbPPArray<arb_value_type>(get_width(), (arb_value_type*)pp->random_numbers[idx], true).to_numpy(); }
ArbIonState PP::ions(size_t idx) {
    if (!pp->ion_states) ERROR("empty ion_states");
    if (idx >= mech->ions.size()) ERROR("param out of range");
    return ArbIonState(get_width(), &pp->ion_states[idx]); }

std::vector<std::shared_ptr<ArbMech>> mechs;

static void init(arb_mechanism_ppack* pp) {
    // i have absolutely NO idea why, but mechanism_id's are assigned
    // in reverse order??
    // Ah so we can not actuall use this it's 
    //   "Unique ID for this mechanism on this cell group"
    // So we need to store in a global I assme
    // int idx = mechs.size() - pp->mechanism_id - 1;
    int idx = (int)pp->globals[0];
    mechs.at(idx)->init_handler(PP(pp, mechs.at(idx)));
}
static void advance_state(arb_mechanism_ppack* pp) {
    int idx = (int)pp->globals[0];
    mechs.at(idx)->advance_state_handler(PP(pp, mechs.at(idx)));
}
static void compute_currents(arb_mechanism_ppack* pp) {
    int idx = (int)pp->globals[0];
    mechs.at(idx)->compute_currents_handler(PP(pp, mechs.at(idx)));
}
static void write_ions(arb_mechanism_ppack* pp) {
    int idx = (int)pp->globals[0];
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
    // we writing against load_catalogue in arbor/mechcat.cpp, see top of file
    frozen_check(true);
    static size_t call_counter = 0;
    // this is ugly, but we know this function is called in a loop
    // so in that way we find out which mechanism we are generating for
    if (call_counter >= mechs.size()) ERROR("called > size(mechs) times; called load_catalogue twice?");
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
        if (mech->globals.at(0).default_value != 123456789.) {
            ERROR("something went wrong with the global idx");
        }
        mech->globals.at(0).default_value = (arb_value_type)mechs.size();
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
        .def_property_readonly("nwidth", &PP::get_nwidth)
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
        .def_property_readonly("current_density", [](ArbIonState & s) { return ArbPPArray<arb_value_type>(s.size_of_data_arrays, s.raw->current_density).to_numpy(); })
        .def_property_readonly("conductivity", [](ArbIonState & s) { return ArbPPArray<arb_value_type>(s.size_of_data_arrays, s.raw->conductivity).to_numpy(); })
        .def_property_readonly("reversal_potential", [](ArbIonState & s) { return ArbPPArray<arb_value_type>(s.size_of_data_arrays, s.raw->reversal_potential).to_numpy(); })
        .def_property_readonly("internal_concentration", [](ArbIonState & s) { return ArbPPArray<arb_value_type>(s.size_of_data_arrays, s.raw->internal_concentration).to_numpy(); })
        .def_property_readonly("external_concentration", [](ArbIonState & s) { return ArbPPArray<arb_value_type>(s.size_of_data_arrays, s.raw->external_concentration).to_numpy(); })
        .def_property_readonly("diffusive_concentration", [](ArbIonState & s) { return ArbPPArray<arb_value_type>(s.size_of_data_arrays, s.raw->diffusive_concentration).to_numpy(); })
        .def_property_readonly("ionic_charge", [](ArbIonState & s) { return ArbPPArray<arb_value_type>(s.size_of_data_arrays, s.raw->ionic_charge).to_numpy(); })
        .def_property_readonly("index", [](ArbIonState & s) { return ArbPPArray<arb_index_type>(s.size_of_index_array, s.raw->index).to_numpy(); })
        ;
#ifdef VERSION_INFO
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
