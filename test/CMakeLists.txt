
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(realtime_asr_test)


set(CMAKE_BUILD_TYPE Debug)
add_compile_options(-g)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

# Get the parent directory 
get_filename_component(MY_PROJECT_ROOT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
message("The cmake current source directory is: ${CMAKE_CURRENT_SOURCE_DIR}")
message("The project root directory is: ${MY_PROJECT_ROOT_DIRECTORY}")

# Set paths to help find pkgs
set(TORCH_PREFIX_PATH "/opt/libtorch")
set(PKG_CONFIG_EXECUTABLE "/usr/bin/pkg-config")

find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PORTAUDIO REQUIRED IMPORTED_TARGET portaudio-2.0)
find_package(Torch REQUIRED PATHS ${TORCH_PREFIX_PATH} NO_DEFAULT_PATH)
find_package(kenlm REQUIRED)

message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
get_cmake_property(_variable_names VARIABLES)




include_directories(${MY_PROJECT_ROOT_DIRECTORY}/include)


# create an exceutable target 
add_executable(streamHandlerTest ${CMAKE_CURRENT_SOURCE_DIR}/utils/test_stream_handler.cpp 
                                 ${MY_PROJECT_ROOT_DIRECTORY}/src/utils/stream_handler.cpp)
add_executable(scriptModelTest   ${CMAKE_CURRENT_SOURCE_DIR}/models/test_torch_script_model.cpp
                                 ${MY_PROJECT_ROOT_DIRECTORY}/src/models/torch_script_model.cpp)
add_executable(greedyDecoderTest ${CMAKE_CURRENT_SOURCE_DIR}/decoders/test_greedy_decoder.cpp
                                 ${MY_PROJECT_ROOT_DIRECTORY}/src/decoders/greedy_decoder.cpp)
add_executable(ngramsModelTest   ${CMAKE_CURRENT_SOURCE_DIR}/models/test_ngrams_model.cpp
                                 ${MY_PROJECT_ROOT_DIRECTORY}/src/models/ngrams_model.cpp)
# # link target dependencies
target_link_libraries(streamHandlerTest 
    GTest::gtest_main
    glog::glog
    PkgConfig::PORTAUDIO
    )
target_link_libraries(scriptModelTest   
    GTest::gtest_main
    ${TORCH_LIBRARIES}
    glog::glog
    )
target_link_libraries(greedyDecoderTest
    GTest::gtest_main
    ${TORCH_LIBRARIES}
    glog::glog)
target_link_libraries(ngramsModelTest
    GTest::gtest_main
    kenlm::kenlm
    glog::glog)