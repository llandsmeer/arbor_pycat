cmake_minimum_required(VERSION 3.15...3.26)

set(CMAKE_CXX_STANDARD 17)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES C CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
    import os, arbor
    print(os.path.join(arbor.__path__[0], 'include'))"
  OUTPUT_VARIABLE arbor_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)

python_add_library(_core MODULE
    main.cpp
    WITH_SOABI)

include_directories(${arbor_INCLUDE_DIR})
target_link_libraries(_core PRIVATE pybind11::headers)
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS _core DESTINATION arbor_pycat)
