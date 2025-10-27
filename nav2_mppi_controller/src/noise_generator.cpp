// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "nav2_mppi_controller/tools/noise_generator.hpp"

#include <memory>
#include <mutex>

namespace mppi
{

void NoiseGenerator::initialize(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & node,
  const std::string & name,
  ParametersHandler * param_handler)
{
  if (auto locked_node = node.lock()) {
    this->logger_ = std::make_unique<rclcpp::Logger>(locked_node->get_logger());
  } else {
    throw std::runtime_error("Failed to lock node for logger initialization.");
  }
  bool regenerate_noises;
  auto getParam = param_handler->getParamGetter(name);
  getParam(regenerate_noises, "regenerate_noises", false);


  // Create publishers for monitoring effective values
  if (auto locked_node = node.lock()) {
    wz_std_pub_ = locked_node->create_publisher<std_msgs::msg::Float32>(
      "/navigation/mppi/wz_std", rclcpp::QoS(1).best_effort());
    // wz_max_pub_ = locked_node->create_publisher<std_msgs::msg::Float32>(
    //   "/navigation/monitor/mppi/wz_max", rclcpp::QoS(1).best_effort());
    current_speed_pub_ = locked_node->create_publisher<std_msgs::msg::Float32>(
      "/navigation/mppi/current_speed", rclcpp::QoS(1).best_effort());
  } else {
    throw std::runtime_error("Failed to lock node for publisher creation.");
  }

  if (regenerate_noises) {
    noise_thread_ = std::make_unique<std::thread>(std::bind(&NoiseGenerator::noiseThread, this));
  } else {
    noise_thread_.reset();
  }

  active_ = true;
}

void NoiseGenerator::shutdown()
{
  active_ = false;
  ready_ = true;
  noise_cond_.notify_all();
  if (noise_thread_ != nullptr) {
    if (noise_thread_->joinable()) {
      noise_thread_->join();
    }
    noise_thread_.reset();
  }
}

void NoiseGenerator::generateNextNoises()
{
  // Trigger the thread to run in parallel to this iteration
  // to generate the next iteration's noises (if applicable).
  {
    std::unique_lock<std::mutex> guard(noise_lock_);
    ready_ = true;
  }
  noise_cond_.notify_all();
}

void NoiseGenerator::setNoisedControls(
  models::State & state,
  const models::ControlSequence & control_sequence)
{
  std::unique_lock<std::mutex> guard(noise_lock_);

  computeAdaptiveStds(state);

  state.cvx = noises_vx_.rowwise() + control_sequence.vx.transpose();
  state.cvy = noises_vy_.rowwise() + control_sequence.vy.transpose();
  state.cwz = noises_wz_.rowwise() + control_sequence.wz.transpose();
}

void NoiseGenerator::computeAdaptiveStds(const models::State & state)
{
  auto & s = settings_;

  // Calculate current speed first
  float current_speed;
  if (is_holonomic_) {
    const auto vx = static_cast<float>(state.speed.linear.x);
    const auto vy = static_cast<float>(state.speed.linear.y);
    current_speed = hypotf(vx, vy);
  } else {
    current_speed = std::fabs(static_cast<float>(state.speed.linear.x));
  }
  

  // Should we apply decay function? or Any constraint is invalid?
  if (s.advanced_constraints.wz_std_decay_strength <= 0.0f || !validateWzStdDecayConstraints()) {
    wz_std_adaptive = s.sampling_std.wz;  // skip calculation
  } else {

    static const float e = std::exp(1.0f);
    const float decayed_wz_std =
      (s.sampling_std.wz - s.advanced_constraints.wz_std_decay_to) *
      powf(e, -1.0f * s.advanced_constraints.wz_std_decay_strength * current_speed) +
      s.advanced_constraints.wz_std_decay_to;

    wz_std_adaptive = decayed_wz_std;
  }
  
  // Store current speed for monitoring
  current_speed_ = current_speed;
}

bool NoiseGenerator::validateWzStdDecayConstraints() const
{
  const models::AdvancedConstraints & c = settings_.advanced_constraints;
  // Assume valid if angular decay is disabled
  if (c.wz_std_decay_strength <= 0.0f) {
    return true;  // valid
  }

  if (c.wz_std_decay_to < 0.0f || c.wz_std_decay_to > settings_.sampling_std.wz) {
    return false;
  }
  return true;
}

float NoiseGenerator::getWzStdAdaptive() const
{
  return wz_std_adaptive;
}

void NoiseGenerator::reset(const mppi::models::OptimizerSettings & settings, bool is_holonomic)
{
  // Recompute the noises on reset, initialization, and fallback
  {
    std::unique_lock<std::mutex> guard(noise_lock_);
    // Copy settings and holonomic info after lock is acquired,
    // otherwise we may encounter concurrent access errors
    settings_ = settings;
    is_holonomic_ = is_holonomic;
    // reset initial adaptive value to parameterized value
    wz_std_adaptive = settings_.sampling_std.wz;
    current_speed_ = 0.0f;
    noises_vx_.setZero(settings_.batch_size, settings_.time_steps);
    noises_vy_.setZero(settings_.batch_size, settings_.time_steps);
    noises_wz_.setZero(settings_.batch_size, settings_.time_steps);

    // Validate decay function, print warning message if decay_to is out of bounds
    if (!validateWzStdDecayConstraints()) {
      RCLCPP_WARN_STREAM(
        *logger_, "wz_std_decay_to must be between 0 and wz_std. wz: "
                    << std::to_string(settings_.constraints.wz) << ", wz_std_decay_to: "
                    << std::to_string(settings_.advanced_constraints.wz_std_decay_to));
    }
    ready_ = true;
  }  // release the lock


  if (noise_thread_ != nullptr) {  // if regenerate_noises_ == true, then noise_thread_ is non-null
    noise_cond_.notify_all();
  } else {
    generateNoisedControls();
  }
}

void NoiseGenerator::noiseThread()
{
  do {
    std::unique_lock<std::mutex> guard(noise_lock_);
    noise_cond_.wait(guard, [this]() {return ready_;});
    ready_ = false;
    generateNoisedControls();
  } while (active_);
}

void NoiseGenerator::generateNoisedControls()
{
  auto & s = settings_;
  publishEffectiveValues();
  auto ndistribution_vx = std::normal_distribution(0.0f, settings_.sampling_std.vx);
  auto ndistribution_wz = std::normal_distribution(0.0f, wz_std_adaptive);
  auto ndistribution_vy = std::normal_distribution(0.0f, settings_.sampling_std.vy);
  noises_vx_ = Eigen::ArrayXXf::NullaryExpr(
    s.batch_size, s.time_steps, [&]() {return ndistribution_vx(generator_);});
  noises_wz_ = Eigen::ArrayXXf::NullaryExpr(
    s.batch_size, s.time_steps, [&]() {return ndistribution_wz(generator_);});
  if (is_holonomic_) {
    noises_vy_ = Eigen::ArrayXXf::NullaryExpr(
      s.batch_size, s.time_steps, [&]() {return ndistribution_vy(generator_);});
  }
}

void NoiseGenerator::publishEffectiveValues() const
{
  if (wz_std_pub_ && current_speed_pub_) {
    auto wz_std_msg = std::make_unique<std_msgs::msg::Float32>();
    // auto wz_max_msg = std::make_unique<std_msgs::msg::Float32>();
    auto current_speed_msg = std::make_unique<std_msgs::msg::Float32>();
    
    wz_std_msg->data = wz_std_adaptive;
    // wz_max_msg->data = settings_.base_constraints.wz;
    current_speed_msg->data = current_speed_;
    
    wz_std_pub_->publish(*wz_std_msg);
    // wz_max_pub_->publish(*wz_max_msg);
    current_speed_pub_->publish(*current_speed_msg);
  }
}
}  // namespace mppi
