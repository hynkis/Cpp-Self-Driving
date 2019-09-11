cmake_minimum_required(VERSION 2.8)
project(mpc_dynamics)

set (CMAKE_CXX_STANDARD 11)

find_package(osqp REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(mpc_dynamics src/main.cpp)


target_link_libraries(mpc_dynamics osqp::osqp osqp::osqpstatic Eigen3::Eigen)
