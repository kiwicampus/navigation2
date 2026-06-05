# Iron compatibility shim.
# nav2_ros_common was written for Jazzy+, where bondcpp exports the namespaced
# imported target `bondcpp::bondcpp`. ROS 2 Iron's bondcpp only exports the old
# ament-style variables (bondcpp_LIBRARIES / bondcpp_INCLUDE_DIRS) and no such
# target, which makes both this package and its consumers fail at link time.
# Recreate the target from those variables when it is missing. The if(NOT TARGET)
# guard makes this a no-op on Jazzy+ where the real target already exists.
if(NOT TARGET bondcpp::bondcpp)
  find_package(bondcpp QUIET)
  add_library(bondcpp::bondcpp INTERFACE IMPORTED)
  set_target_properties(bondcpp::bondcpp PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${bondcpp_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${bondcpp_LIBRARIES}")
endif()
