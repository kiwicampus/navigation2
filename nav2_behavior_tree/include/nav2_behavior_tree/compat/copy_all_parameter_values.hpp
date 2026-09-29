#pragma once

// rclcpp::copy_all_parameter_values() (contributed by Open Navigation LLC) isn't present in
// ROS 2 Humble's rclcpp, only in newer distros; BtActionServer::on_configure() below calls it
// unconditionally. __has_include lets this shim disable itself the moment the real header
// exists, so it needs no manual removal once the deployment target moves off Humble.
#if __has_include(<rclcpp/copy_all_parameter_values.hpp>)
#include <rclcpp/copy_all_parameter_values.hpp>
#else

#include <string>
#include <vector>

#include "rcl_interfaces/srv/list_parameters.hpp"
#include "rcl_interfaces/msg/parameter_descriptor.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"

#include "rclcpp/parameter.hpp"
#include "rclcpp/logger.hpp"
#include "rclcpp/logging.hpp"

namespace rclcpp
{

// Verbatim port of the upstream implementation (rclcpp/copy_all_parameter_values.hpp, Jazzy+).
template<typename NodeT1, typename NodeT2>
void
copy_all_parameter_values(
  const NodeT1 & source, const NodeT2 & destination, const bool override_existing_params = false)
{
  using Parameters = std::vector<rclcpp::Parameter>;
  using Descriptions = std::vector<rcl_interfaces::msg::ParameterDescriptor>;
  auto source_params = source->get_node_parameters_interface();
  auto dest_params = destination->get_node_parameters_interface();
  rclcpp::Logger logger = destination->get_node_logging_interface()->get_logger();

  std::vector<std::string> param_names = source_params->list_parameters({}, 0).names;
  Parameters params = source_params->get_parameters(param_names);
  Descriptions descriptions = source_params->describe_parameters(param_names);

  for (unsigned int idx = 0; idx != params.size(); idx++) {
    if (!dest_params->has_parameter(params[idx].get_name())) {
      dest_params->declare_parameter(
        params[idx].get_name(), params[idx].get_parameter_value(), descriptions[idx]);
    } else if (override_existing_params) {
      try {
        rcl_interfaces::msg::SetParametersResult result =
          dest_params->set_parameters_atomically({params[idx]});
        if (!result.successful) {
          RCLCPP_WARN(
            logger, "Unable to set parameter (%s): %s!",
            params[idx].get_name().c_str(), result.reason.c_str());
        }
      } catch (const rclcpp::exceptions::InvalidParameterTypeException & e) {
        RCLCPP_WARN(
          logger, "Unable to set parameter (%s): incompatable parameter type (%s)!",
          params[idx].get_name().c_str(), e.what());
      }
    }
  }
}

}  // namespace rclcpp

#endif  // __has_include(<rclcpp/copy_all_parameter_values.hpp>)
