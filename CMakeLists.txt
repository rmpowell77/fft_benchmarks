cmake_minimum_required(VERSION 3.23)

set (CMAKE_CXX_STANDARD 20)

find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)
include_directories(PkgConfig::FFTW)
link_libraries     (PkgConfig::FFTW)

project(
  fft_benchmark
  VERSION 1.0
  LANGUAGES CXX)

add_executable(fft_benchmark fft_benchmark.cpp)


