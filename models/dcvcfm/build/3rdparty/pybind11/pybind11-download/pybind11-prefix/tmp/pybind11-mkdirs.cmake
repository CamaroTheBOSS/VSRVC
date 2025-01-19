# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "D:/Code/DCVC/DCVC-FM/src/build/3rdparty/pybind11/pybind11-src"
  "D:/Code/DCVC/DCVC-FM/src/build/3rdparty/pybind11/pybind11-build"
  "D:/Code/DCVC/DCVC-FM/src/build/3rdparty/pybind11/pybind11-download/pybind11-prefix"
  "D:/Code/DCVC/DCVC-FM/src/build/3rdparty/pybind11/pybind11-download/pybind11-prefix/tmp"
  "D:/Code/DCVC/DCVC-FM/src/build/3rdparty/pybind11/pybind11-download/pybind11-prefix/src/pybind11-stamp"
  "D:/Code/DCVC/DCVC-FM/src/build/3rdparty/pybind11/pybind11-download/pybind11-prefix/src"
  "D:/Code/DCVC/DCVC-FM/src/build/3rdparty/pybind11/pybind11-download/pybind11-prefix/src/pybind11-stamp"
)

set(configSubDirs Debug;Release;MinSizeRel;RelWithDebInfo)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "D:/Code/DCVC/DCVC-FM/src/build/3rdparty/pybind11/pybind11-download/pybind11-prefix/src/pybind11-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "D:/Code/DCVC/DCVC-FM/src/build/3rdparty/pybind11/pybind11-download/pybind11-prefix/src/pybind11-stamp${cfgdir}") # cfgdir has leading slash
endif()
