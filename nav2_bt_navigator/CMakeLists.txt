cmake_minimum_required(VERSION 3.8)
project(nav2_bt_navigator)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(backward_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav2_behavior_tree REQUIRED)
find_package(nav2_common REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(nav2_util REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(nav2_ros_common REQUIRED)

nav2_package()

set(executable_name bt_navigator)

set(library_name ${executable_name}_core)

add_library(${library_name} SHARED
  src/bt_navigator.cpp
)
target_include_directories(${library_name}
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
)
target_link_libraries(${library_name} PUBLIC
  nav2_core::nav2_core
  nav2_util::nav2_util_core
  nav2_ros_common::nav2_ros_common
  pluginlib::pluginlib
  rclcpp::rclcpp
  rclcpp_action::rclcpp_action
  tf2_ros::tf2_ros
)
target_link_libraries(${library_name} PRIVATE
  nav2_behavior_tree::nav2_behavior_tree
  rclcpp_components::component
  rclcpp_lifecycle::rclcpp_lifecycle
)

add_executable(${executable_name}
  src/main.cpp
)
target_include_directories(${executable_name}
  PRIVATE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
)
target_link_libraries(${executable_name} PRIVATE
  ${library_name}
  rclcpp::rclcpp
  nav2_ros_common::nav2_ros_common
)

# nav2_navigate_to_pose_navigator / nav2_navigate_through_poses (src/navigators/*.cpp) are not
# built here: both instantiate BehaviorTreeNavigator<ActionT>::on_configure(), which calls
# rclcpp::copy_all_parameter_values(), unavailable in ROS 2 Humble's rclcpp (the actual robot
# deployment target). This repo only needs the generic BtNavigator host plus its own
# manipulation::ManipulationNavigator plugin, not nav2's stock navigators.
rclcpp_components_register_nodes(${library_name} "nav2_bt_navigator::BtNavigator")

install(TARGETS ${library_name}
  EXPORT export_${library_name}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS ${executable_name}
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

install(DIRECTORY behavior_trees DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include/${PROJECT_NAME})
ament_export_libraries(${library_name})
ament_export_dependencies(geometry_msgs nav2_core nav2_msgs nav2_util nav2_ros_common nav_msgs pluginlib rclcpp rclcpp_action tf2_ros)
ament_export_targets(export_${library_name})
ament_package()
