cmake_minimum_required(VERSION 2.8)
project(osqp_eigen_test)

set (CMAKE_CXX_STANDARD 11)

find_package(osqp REQUIRED)
find_package(Eigen3 3.2 REQUIRED NO_MODULE)

add_executable(osqp_eigen_test src/main.cpp)


target_link_libraries(osqp_eigen_test osqp::osqp osqp::osqpstatic Eigen3::Eigen)
