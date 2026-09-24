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

#ifndef NAV2_MPPI_CONTROLLER__MODELS__CONSTRAINTS_HPP_
#define NAV2_MPPI_CONTROLLER__MODELS__CONSTRAINTS_HPP_

namespace mppi::models
{

/**
 * @struct mppi::models::ControlConstraints
 * @brief Constraints on control
 */
struct ControlConstraints
{
  float vx_max;
  float vx_min;
  float vy;
  float wz;
  float ax_max;
  float ax_min;
  float ay_min;
  float ay_max;
  float az_max;
};

/**
 * @struct mppi::models::AdvancedConstraints
 * @brief Speed dependent sampling parameters
 */
struct AdvancedConstraints
{
  /**
   * @brief Strength of the wz_std decay as a function of the robot linear speed.
   * High wz_std at high speed produces laterally spread trajectories that make the
   * robot oscillate and prefer slower samples. Decaying wz_std as speed increases
   * keeps maneuverability at low speed and stability at high speed.
   * <pre>f(v) = (wz_std - wz_std_decay_to) * e^(-wz_std_decay_strength * v) + wz_std_decay_to</pre>
   * Default: -1.0 (disabled)
   */
  float wz_std_decay_strength;

  /**
   * @brief Target wz_std value while linear speed goes to infinity.
   * Must be between 0 and wz_std. Has no effect if wz_std_decay_strength <= 0.0
   * Default: 0.0
   */
  float wz_std_decay_to;
};

/**
 * @struct mppi::models::SamplingStd
 * @brief Noise parameters for sampling trajectories
 */
struct SamplingStd
{
  float vx;
  float vy;
  float wz;
};

}  // namespace mppi::models

#endif  // NAV2_MPPI_CONTROLLER__MODELS__CONSTRAINTS_HPP_
