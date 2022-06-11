cmake_minimum_required(VERSION 3.23)

set (CMAKE_CXX_STANDARD 20)

project(
  fft_benchmark
  VERSION 1.0
  LANGUAGES CXX)

add_executable(fft_benchmark fft_benchmark.cpp)


