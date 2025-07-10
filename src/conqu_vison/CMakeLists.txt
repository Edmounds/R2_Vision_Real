cmake_minimum_required(VERSION 3.0.2)
project(conqu_vison)

## 查找依赖包
find_package(catkin REQUIRED COMPONENTS
  pcl_ros
  roscpp
  sensor_msgs
  std_msgs
  geometry_msgs
  message_generation
  visualization_msgs
  tf2
  tf2_geometry_msgs
)

# ## 声明消息文件
# add_message_files(
#   FILES
#   HoopCoordinates.msg
#   HoopDetection.msg
# )

## 生成消息
generate_messages(
  DEPENDENCIES
  sensor_msgs
  std_msgs
  geometry_msgs
)

## catkin包配置
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES coordinate_calculator_lib hoop_detector_lib
  CATKIN_DEPENDS pcl_ros roscpp sensor_msgs std_msgs geometry_msgs message_runtime visualization_msgs tf2 tf2_geometry_msgs
)

## 包含头文件目录
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

## 将coordinate_calculator和hoop_detector构建为库
add_library(coordinate_calculator_lib
  src/coordinate_calculator.cpp
)

add_library(hoop_detector_lib
  src/hoop_detector.cpp
)

## 添加依赖关系
add_dependencies(coordinate_calculator_lib ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(hoop_detector_lib ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## 链接库到相应的依赖库
target_link_libraries(coordinate_calculator_lib
  ${catkin_LIBRARIES}
)

target_link_libraries(hoop_detector_lib
  ${catkin_LIBRARIES}
)

## 声明主可执行文件
add_executable(main_node src/main_node.cpp)

## 添加主可执行文件的依赖关系
add_dependencies(main_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## 将库链接到主可执行文件
target_link_libraries(main_node
  coordinate_calculator_lib
  hoop_detector_lib
  ${catkin_LIBRARIES}
)
