cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_BUILD_TYPE Debug)
add_compile_options(-g)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# project related 
project(realtime_asr_src)
get_filename_component(MY_PROJECT_ROOT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
message("Project Root: ${MY_PROJECT_ROOT_DIRECTORY}")

# Set paths to help find pkgs
set(TORCH_PREFIX_PATH "/opt/libtorch")
set(PKG_CONFIG_EXECUTABLE "/usr/bin/pkg-config")

# Find pkgs 
find_package(Torch REQUIRED PATHS ${TORCH_PREFIX_PATH} NO_DEFAULT_PATH)
message(STATUS "Found Torch library at: ${TORCH_LIBRARIES}")
find_package(glog REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PORTAUDIO REQUIRED IMPORTED_TARGET portaudio-2.0)
find_library(OpenFst NAMES fst oepnfst REQUIRED)
message(STATUS "OpenFst found at: ${OpenFst}")
if (${OpenFst-NOTFOUND})
    message ("Failed to find OpenFST")
endif()
find_package(kenlm REQUIRED)
message(STATUS "found kenlm using kenlm-FOUND: ${kenlm_FOUND}")
message(STATUS "kenlm libraries using kenlm-LIBRARIES: ${kenlm-INCLUDE_DIRS}")

# Add local include directory 
include_directories(${MY_PROJECT_ROOT_DIRECTORY}/include)

# Create Exe 1
add_executable(test ${CMAKE_CURRENT_SOURCE_DIR}/decoders/test.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/decoders/ctc_decoder.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/decoders/lexicon.cpp 
                    ${CMAKE_CURRENT_SOURCE_DIR}/models/torch_script_model.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/models/ngrams_model.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/utils/my_utils.cpp
                    )

target_link_libraries(test PRIVATE glog::glog
                           PRIVATE ${TORCH_LIBRARIES} 
                           PRIVATE ${OpenFst}
                           PRIVATE kenlm::kenlm)


# Create Exe 2
add_executable(setup ${CMAKE_CURRENT_SOURCE_DIR}/decoders/setup.cpp
                     ${CMAKE_CURRENT_SOURCE_DIR}/decoders/lexicon.cpp
                     )

target_link_libraries(setup PRIVATE glog::glog
                           PRIVATE ${OpenFst})


# add_executable(test_sree3 ${CMAKE_CURRENT_SOURCE_DIR}/utils/test_sree3.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/decoders/lexicon.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/utils/my_utils.cpp)

# target_link_libraries(test_sree3 kenlm::kenlm glog::glog ${OpenFst})

# add_executable(test_asra3 ${CMAKE_CURRENT_SOURCE_DIR}/utils/test_asra3.cpp
# ${CMAKE_CURRENT_SOURCE_DIR}/models/torch_script_model.cpp
# ${CMAKE_CURRENT_SOURCE_DIR}/utils/my_utils.cpp)

# target_link_libraries(test_asra3 kenlm::kenlm glog::glog ${OpenFst} ${TORCH_LIBRARIES})