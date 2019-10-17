cmake_minimum_required (VERSION 3.0)
project (preCICE-Eigen VERSION 0.0.1 LANGUAGES CXX)


# Enforce a build type
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
message(STATUS "Build configuration: " ${CMAKE_BUILD_TYPE})

add_executable(eigen_precice src/main.cpp)

target_compile_features(eigen_precice PRIVATE cxx_std_11)

target_include_directories(eigen_precice PRIVATE ${PRECICE_INCLUDES})
target_link_libraries(eigen_precice PRIVATE ${PRECICE_LIBRARIES})
target_include_directories(eigen_precice PRIVATE src/)

find_package(Eigen3 REQUIRED)
if(${EIGEN3_FOUND})
  target_link_libraries(eigen_precice PRIVATE Eigen3::Eigen)
else()
  MESSAGE(WARNING "Eigen3 not found!")
endif()

find_package(precice REQUIRED CONFIG)
target_link_libraries(eigen_precice PRIVATE precice::precice)
