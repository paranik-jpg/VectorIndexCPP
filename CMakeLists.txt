cmake_minimum_required(VERSION 3.20)
# Sets the name of the project.
project(VectorIndexCPP CXX)

set(CMAKE_CXX_STANDARD 20)
# Guarantees that C++20 is strictly required.
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Automatically generates compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# This allows any .cpp file in our project to
# write #include "vector_index.hpp" directly
# without needing relative paths like ../include/vector_index.hpp.
include_directories(include)

# Binary name & file to compile
add_executable(test_step1 tests/test_step1.cpp)
