cmake_minimum_required(VERSION 2.8)
project(eigen_sparse_test)

set (CMAKE_CXX_STANDARD 11)

find_package(osqp REQUIRED)
find_package(Eigen3 3.2 REQUIRED NO_MODULE)

add_executable(eigen_sparse_test src/main.cpp)


target_link_libraries(eigen_sparse_test osqp::osqp osqp::osqpstatic Eigen3::Eigen)
