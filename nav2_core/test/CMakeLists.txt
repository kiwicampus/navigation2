# NavigatorMuxer test
nav2_add_gtest(test_navigator_muxer test_navigator_muxer.cpp)
target_link_libraries(test_navigator_muxer
  nav2_core
)

# BehaviorTreeNavigator test intentionally not built: it instantiates
# BehaviorTreeNavigator<ActionT>::on_configure(), which calls rclcpp::copy_all_parameter_values(),
# unavailable in ROS 2 Humble's rclcpp (the actual robot deployment target).
