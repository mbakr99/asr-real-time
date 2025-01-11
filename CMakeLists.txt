cmake_minimum_required(VERSION 3.10)
project(record_live)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set paths to help find Python 3.10
set(Python3_ROOT_DIR "/usr")
set(Python3_EXECUTABLE "/usr/bin/python3.10")

# Find Python with all components
find_package(Python3 3.10 EXACT REQUIRED 
    COMPONENTS Interpreter Development NumPy) #Development.Module


# Find PortAudio
find_library(PORTAUDIO_LIB portaudio REQUIRED)

# Print debug information
message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3_NumPy_INCLUDE_DIRS: ${Python3_NumPy_INCLUDE_DIRS}")

# Include directories
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})

# Add the source file
add_executable(test1 src/record_live.cpp)

# Link libraries
target_link_libraries(test1 Python3::Python ${PORTAUDIO_LIB})

# Include directoris 
target_include_directories(test1 PRIVATE ${CMAKE_SOURCE_DIR}/include)
message(STATUS "Project Directory: ${CMAKE_SOURCE_DIR}")