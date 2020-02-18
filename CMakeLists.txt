cmake_minimum_required(VERSION 3.12)

project(dtrt VERSION 0.0.1 DESCRIPTION "Differentiable Radiative Transfer")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

if(WIN32)
    find_package(PythonLibs 3.6 COMPONENTS Development REQUIRED)
    add_subdirectory(pybind11)
else()
    find_package(Python 3.6 COMPONENTS Development REQUIRED)
endif()

find_package(Embree REQUIRED)
if(NOT EMBREE_FOUND)
    add_subdirectory(embree)
endif()

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(${Python_INCLUDE_DIRS})
include_directories(${EMBREE_INCLUDE_PATH})
include_directories(include/)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2 /O2 /Z7 /wd4244 /wd4267 /wd4305 /wd4996")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -Wno-ignored-attributes -Wno-int-in-bool-context -Wall -g -O3 -fvisibility=hidden")
endif()

set(SRCS include/config.h
         include/fwd.h
         include/ptr.h
         include/stats.h
         src/stats.cpp
         include/math_func.h
         src/math_func.cpp
         include/line_clip.h
         include/utils.h
         src/utils.cpp
         include/frame.h
         include/frameAD.h
         include/ray.h
         include/rayAD.h
         include/sampler.h
         src/sampler.cpp
         include/camera.h
         src/camera.cpp
         include/phase.h
         include/shape.h
         src/shape.cpp
         include/intersection.h
         include/intersectionAD.h
         include/edge_manager.h
         src/edge_manager/edge_manager.cpp
         src/edge_manager/bruteforce.h
         src/edge_manager/bruteforce.cpp
         include/scene.h
         src/scene.cpp
         include/emitter.h
         src/emitter/area.h
         src/emitter/area.cpp
         src/emitter/area2.h
         src/emitter/area2.cpp
         include/bsdf.h
         src/bsdf.cpp
         src/bsdf/null.h
         src/bsdf/null.cpp
         src/bsdf/diffuse.h
         src/bsdf/diffuse.cpp
         src/bsdf/phong.h
         src/bsdf/phong.cpp
         src/bsdf/microfacet.h
         src/bsdf/roughdielectric.h
         src/bsdf/roughdielectric.cpp
         include/medium.h
         src/medium/homogeneous.h
         src/medium/homogeneous.cpp
         src/medium/gridvolume.h
         src/medium/heterogeneous.h
         src/medium/heterogeneous.cpp
         include/integrator.h
         src/integrator/direct.h
         src/integrator/direct.cpp
         src/integrator/path.h
         src/integrator/path.cpp
         src/integrator/volpath_simple.h
         src/integrator/volpath_simple.cpp
         src/integrator/volpath.h
         src/integrator/volpath.cpp
         src/integrator/ptracer.h
         src/integrator/ptracer.cpp
         include/integratorAD.h
         src/integrator/differential/integratorAD.cpp
         src/integrator/differential/directAD.h
         src/integrator/differential/directAD.cpp
         src/integrator/differential/pathAD.h
         src/integrator/differential/pathAD.cpp
         src/integrator/differential/volpathAD.h
         src/integrator/differential/volpathAD.cpp
         src/dtrt.cpp
         )

if(APPLE)
    set(DYNAMIC_LOOKUP "-undefined dynamic_lookup")
endif()
if (WIN32)
    pybind11_add_module(dtrt SHARED  ${SRCS} )
endif()

if (NOT WIN32)
    add_library(dtrt MODULE ${SRCS})
    # The "-undefined dynamic_lookup" is a hack for systems with
    # multiple Python installed. If we link a particular Python version
    # here, and we import it with a different Python version later.
    # likely a segmentation fault.
    # The solution for Linux/Mac OS machines, as mentioned in
    # https://github.com/pybind/pybind11/blob/master/tools/pybind11Tools.cmake
    # is to not link against Python library at all and resolve the symbols
    # at compile time.
    target_link_libraries(dtrt
        ${EMBREE_LIBRARY}
        ${DYNAMIC_LOOKUP})
else()
    target_link_libraries(dtrt
    PRIVATE
        ${EMBREE_LIBRARY}
        )
endif()

set_target_properties(dtrt PROPERTIES PREFIX "")

execute_process(
    COMMAND "python3" -c "if True:
        from distutils import sysconfig as sc
        print(sc.get_python_lib())"
    OUTPUT_VARIABLE PYTHON_SITE
    OUTPUT_STRIP_TRAILING_WHITESPACE)

install(
    TARGETS dtrt
    LIBRARY DESTINATION lib
    DESTINATION ${PYTHON_SITE})

execute_process (
    COMMAND "python3" -c "from distutils.sysconfig import get_python_version; print(get_python_version())"
    OUTPUT_VARIABLE PYTHON_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)
install(CODE "execute_process(COMMAND python${PYTHON_VERSION} ${CMAKE_CURRENT_SOURCE_DIR}/setup.py install
                              WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})")
