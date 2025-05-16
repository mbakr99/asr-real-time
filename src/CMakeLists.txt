cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_BUILD_TYPE Debug)
add_compile_options(-g)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# user options 
option(SHOW_PROGRESS "Show progress bar" ON)  


# fetch content for more robust and protbale distribution
include(FetchContent)
include(ExternalProject)
include(ProcessorCount)
ProcessorCount(NPROC)
message(STATUS "Number of cores available is: ${NPROC}")
# project related 
project(realtime_asr_src)
get_filename_component(MY_PROJECT_ROOT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
message("Project Root: ${MY_PROJECT_ROOT_DIRECTORY}")


# Set paths to help find pkgs
set(TORCH_PREFIX_PATH "/opt/libtorch")
set(PKG_CONFIG_EXECUTABLE "/usr/bin/pkg-config")


## Find pkgs 
# Find Torch 
find_package(Torch PATHS ${TORCH_PREFIX_PATH} NO_DEFAULT_PATH)
message(STATUS "Found Torch library at: ${TORCH_LIBRARIES}")

if (Torch_FOUND)
    message(STATUS "Torch found")
elseif(SHOW_PROGRESS)
    #set torch installtion variables
    set(LibTorch_VERSION "2.7.0")
    set(LibTorch_URL https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-${LibTorch_VERSION}%2Bcu118.zip)
    set(LibTorch_ZIP "${CMAKE_BINARY_DIR}/libtorch.zip")

    message(STATUS "Downloading LibTorch... (this may take a while)")

    file(
        DOWNLOAD
        ${LibTorch_URL}
        ${LibTorch_ZIP}
        SHOW_PROGRESS
        STATUS DOWNLOAD_STATUS
    )

    # check installtion result
    list(GET DOWNLOAD_STATUS 0 DOWNLOAD_RESULT)
    if (NOT DOWNLOAD_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to download libtorch: ${DOWNLOAD_STATUS}")
    endif()

    # extract files
    set(Torch_DESTINATION ${CMAKE_BINARY_DIR})
    file(ARCHIVE_EXTRACT INPUT ${LibTorch_ZIP} DESTINATION ${Torch_DESTINATION})

    # set torch path 
    set(Torch_PREFIX_PATH ${Torch_DESTINATION}/libtorch CACHE PATH "Torch install prefix")
    find_package(Torch REQUIRED PATHS ${Torch_PREFIX_PATH} NO_DEFAULT_PATH)  
    
else()
    message(STATUS "Torch not found. Installing prebuilt library")

    # set torch installtion variables
    set(LibTorch_VERSION "2.7.0")
    set(LibTorch_URL https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${LibTorch_VERSION}%2Bcpu.zip)
    FetchContent_Declare(
        LibTorch
        URL ${LibTorch_URL}
    )

    # download and extract using fetchcontent 
    FetchContent_Populate(LibTorch)

    # add the package
    set(Torch_PREFIX_PATH ${LibTorch_SOURCE_DIRECTORY} CACHE PATH "Torch install prefix")
    find_package(Torch REQUIRED PATHS ${Torch_PREFIX_PATHS} NO_DEFAULT_PATH)
endif()


# Find glog 
# set(DUM_GLOG_PATH "/usr/local/opt/tmp") Note: this is to be used for testing the robustness of the glog installation 
find_package(glog QUIET) # PATHS DUM_OPENFST_PATH NO_DEFAULT_PATH
if (glog_FOUND)
    message(STATUS "Found glog")
else()
    message(STATUS "glog was not found. Installing and building from source")
    FetchContent_Declare(
        glog
        GIT_REPOSITORY https://github.com/google/glog.git
        GIT_TAG 7b134a5c82c0c0b5698bb6bf7a835b230c5638e4 # rlease v0.7.1
    )
    FetchContent_MakeAvailable(
        glog
    )
endif()


# Find portaudio 
find_package(PkgConfig REQUIRED)
pkg_check_modules(PORTAUDIO QUIET IMPORTED_TARGET portaudio-.0) #2
if (PORTAUDIO_FOUND)
    message(STATUS "Found portaudio")
else()
    message(STATUS "Did not find portaudio. Adding external project")
    # set installation directory 
    set(portaudio_INSTALL_DIR ${CMAKE_BINARY_DIR}/_deps/portaudio_install)

    # install and configure 
    set(portaudio_URL "https://files.portaudio.com/archives/pa_stable_v190700_20210406.tgz")
    ExternalProject_Add(
        portaudio 
        URL ${portaudio_URL}
        CONFIGURE_COMMAND ./configure --prefix=${portaudio_INSTALL_DIR}
        BUILD_COMMAND  make -j${NPROC}
        INSTALL_COMMAND make install
        BUILD_IN_SOURCE 1
    )

    message(STATUS "Installed portaudio files in ${portaudio_INSTALL_DIR}")
    set(ENV{PKG_CONFIG_PATH} "${portaudio_INSTALL_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
    message(STATUS "PKG_CONFIG_PATH: $ENV{PKG_CONFIG_PATH}")
    pkg_check_modules(PORTAUDIO REQUIRED IMPORTED_TARGET portaudio-2.0) #Question: Why does this work? Does usign pkg_check_module force cmake to build portaduio
endif()


# Find openFst
# set(DUM_OPENFST_PATH "/opt/tmp") # Note: this is to be used to test the robustness of instlling openfst 
find_library(OpenFst NAMES fst oepnfst QUIET) # PATHS DUM_OPENFST_PATH NO_DEFAULT_PATH

if (${OpenFst})
    message("Found OpenFst at: ${OpenFst}")
else()
    message(STATUS "No installation of OpenFst was found")
    
    # set installation   
    set(openfst_INSTALL_DIR ${CMAKE_BINARY_DIR}/_deps/openfst_install)
    set(openfst_URL "https://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.8.4.tar.gz")
    ExternalProject_Add(
        openfst
        URL ${openfst_URL}
        PREFIX ${CMAKE_BINARY_DIR}/_deps/openfst
        CONFIGURE_COMMAND <SOURCE_DIR>/configure 
                          --prefix=${openfst_INSTALL_DIR}
                          --enable-static=yes  # Fix 3: Must match what you're linking against
                          --disable-shared
        BUILD_COMMAND make -j${NPROC}
        INSTALL_COMMAND make install 
        BUILD_IN_SOURCE 0  # Fix 4: Build out of source
    )
    
    file(MAKE_DIRECTORY ${openfst_INSTALL_DIR}/include)
    file(MAKE_DIRECTORY ${openfst_INSTALL_DIR}/lib)

    # define a placeholder target 
    add_library(openfst_lib STATIC IMPORTED GLOBAL)
    add_dependencies(openfst_lib openfst)
    set_target_properties(openfst_lib PROPERTIES
        IMPORTED_LOCATION "${openfst_INSTALL_DIR}/lib/libfst.a"
        INTERFACE_INCLUDE_DIRECTORIES "${openfst_INSTALL_DIR}/include"
        INTERFACE_LINK_LIBRARIES "dl"
    )
    
endif()





# Find kenlm
set(dum_kenlm_PATH "/opt/tmp") #FIXME:
find_package(kenlm PATHS ${dum_kenlm_PATH} QUIET NO_DEFAULT_PATH)
if (${kenlm_FOUND})
    message(STATUS "Found kenlm")
else()
    message(STATUS "No installtion of kenlm found. Fetchign content from Git repo")

    set(kenlm_GIT_REPO "https://github.com/kpu/kenlm.git")
    FetchContent_Declare(
        kenlm 
        GIT_REPOSITORY ${kenlm_GIT_REPO}   
    )
    FetchContent_MakeAvailable(kenlm)
endif()





# Add local include directory 
include_directories(${MY_PROJECT_ROOT_DIRECTORY}/include)



add_executable(robust_test ${CMAKE_CURRENT_SOURCE_DIR}/decoders/test_sree3.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/utils/stream_handler.cpp)

target_link_libraries(robust_test PRIVATE openfst_lib
                                  PRIVATE glog::glog
                                  PkgConfig::PORTAUDIO)                          


# Create Exe 1
add_executable(test ${CMAKE_CURRENT_SOURCE_DIR}/decoders/test.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/decoders/ctc_decoder.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/decoders/lexicon.cpp 
                    ${CMAKE_CURRENT_SOURCE_DIR}/models/torch_script_model.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/models/ngrams_model.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/utils/my_utils.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/decoders/beam.cpp
                    )

target_link_libraries(test PRIVATE glog::glog
                           PRIVATE ${TORCH_LIBRARIES} 
                           PRIVATE openfst_lib #FIXME: 
                           PRIVATE kenlm)


# # Create Exe 2
# add_executable(setup ${CMAKE_CURRENT_SOURCE_DIR}/decoders/setup.cpp
#                      ${CMAKE_CURRENT_SOURCE_DIR}/decoders/lexicon.cpp
#                      )

# target_link_libraries(setup PRIVATE glog::glog
#                             PRIVATE openfst_lib #FIXME: 
#                     )


# add_executable(test_sree3 ${CMAKE_CURRENT_SOURCE_DIR}/decoders/test_sree3.cpp
#                           ${CMAKE_CURRENT_SOURCE_DIR}/utils/stream_handler.cpp
#                 )

# target_link_libraries(test_sree3 glog::glog PkgConfig::PORTAUDIO)

