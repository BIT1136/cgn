cmake_minimum_required(VERSION 3.0.2)
project(cgn)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  rospy
  sensor_msgs
  std_msgs
)

catkin_python_setup()

add_service_files(
  FILES
  GenerateGrasps.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  sensor_msgs
  std_msgs
)

catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
