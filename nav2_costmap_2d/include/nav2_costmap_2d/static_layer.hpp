/*********************************************************************
 *
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2008, 2013, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: Eitan Marder-Eppstein
 *         David V. Lu!!
 *********************************************************************/
#ifndef NAV2_COSTMAP_2D__STATIC_LAYER_HPP_
#define NAV2_COSTMAP_2D__STATIC_LAYER_HPP_

#include <mutex>
#include <string>
#include <vector>

#include "map_msgs/msg/occupancy_grid_update.hpp"
#include "message_filters/subscriber.h"
#include "nav2_costmap_2d/costmap_layer.hpp"
#include "nav2_costmap_2d/layered_costmap.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "rclcpp/rclcpp.hpp"
#include "nav2_costmap_2d/footprint.hpp"

namespace nav2_costmap_2d
{

/**
 * @class StaticLayer
 * @brief Takes in a map generated from SLAM to add costs to costmap
 */
class StaticLayer : public CostmapLayer
{
public:
  /**
    * @brief Static Layer constructor
    */
  StaticLayer();
  /**
    * @brief Static Layer destructor
    */
  virtual ~StaticLayer();

  /**
   * @brief Initialization process of layer on startup
   */
  virtual void onInitialize();

  /**
   * @brief Activate this layer
   */
  virtual void activate();
  /**
   * @brief Deactivate this layer
   */
  virtual void deactivate();

  /**
   * @brief Reset this costmap
   */
  virtual void reset();

  /**
   * @brief If clearing operations should be processed on this layer or not
   */
  virtual bool isClearable() {return false;}

  /**
   * @brief Update the bounds of the master costmap by this layer's update dimensions
   * @param robot_x X pose of robot
   * @param robot_y Y pose of robot
   * @param robot_yaw Robot orientation
   * @param min_x X min map coord of the window to update
   * @param min_y Y min map coord of the window to update
   * @param max_x X max map coord of the window to update
   * @param max_y Y max map coord of the window to update
   */
  virtual void updateBounds(
    double robot_x, double robot_y, double robot_yaw, double * min_x,
    double * min_y, double * max_x, double * max_y);

  /**
   * @brief Update the costs in the master costmap in the window
   * @param master_grid The master costmap grid to update
   * @param min_x X min map coord of the window to update
   * @param min_y Y min map coord of the window to update
   * @param max_x X max map coord of the window to update
   * @param max_y Y max map coord of the window to update
   */
  virtual void updateCosts(
    nav2_costmap_2d::Costmap2D & master_grid,
    int min_i, int min_j, int max_i, int max_j);

  /**
   * @brief Match the size of the master costmap
   */
  virtual void matchSize();

protected:
  /**
   * @brief Get parameters of layer
   */
  void getParameters();

  /**
   * @brief Process a new map coming from a topic
   */
  void processMap(const nav_msgs::msg::OccupancyGrid & new_map);

  /**
   * @brief  Callback to update the costmap's map from the map_server
   * @param new_map The map to put into the costmap. The origin of the new
   * map along with its size will determine what parts of the costmap's
   * static map are overwritten.
   */
  void incomingMap(const nav_msgs::msg::OccupancyGrid::SharedPtr new_map);
  /**
   * @brief Callback to update the costmap's map from the map_server (or SLAM)
   * with an update in a particular area of the map
   */
  void incomingUpdate(map_msgs::msg::OccupancyGridUpdate::ConstSharedPtr update);

  /**
   * @brief Interpret the value in the static map given on the topic to
   * convert into costs for the costmap to utilize
   */
  unsigned char interpretValue(unsigned char value);

  /**
   * @brief Cache which cells of the raw map belong to the drivable region
   *
   * Operates on the RAW occupancy values rather than on the interpreted
   * costmap: with `track_unknown_space: false` unknown cells are interpreted
   * as free space, which would otherwise make everything outside the corridor
   * count as drivable and dissolve the boundary instead of shifting it.
   *
   * @param new_map The map to read the drivable region from
   */
  void buildRawFreeMask(const nav_msgs::msg::OccupancyGrid & new_map);

  /**
   * @brief Whether a raw map value belongs to the drivable region
   * @param value Raw occupancy value, as stored in the map message
   */
  inline bool isRawFree(unsigned char value) const
  {
    return value != unknown_cost_value_ && value < lethal_threshold_;
  }

  /**
   * @brief Precompute the disc offsets used to grow the drivable region
   */
  void computeErosionOffsets();

  /**
   * @brief Build the erosion masks for one rectangle of the map
   *
   * Only the cells the costmap window actually reads are eroded, which is
   * ~550x less work than the whole map for the global costmap and ~3700x less
   * for the local one. The rectangle is grown by the ring radius internally so
   * that cells just outside it can still seed growth into it.
   *
   * @param x0 @param y0 @param x1 @param y1 Inclusive rectangle in map cells
   */
  void updateErosionWindow(int x0, int y0, int x1, int y1);

  /**
   * @brief Read a cell of this layer, applying the erosion when active
   * @param mx The x coordinate of the cell in this layer
   * @param my The y coordinate of the cell in this layer
   * @return The (possibly eroded) cost of the cell
   */
  inline unsigned char getStaticCost(unsigned int mx, unsigned int my) const
  {
    if (erosion_active_) {
      const int lx = static_cast<int>(mx) - erosion_window_x_;
      const int ly = static_cast<int>(my) - erosion_window_y_;
      if (lx >= 0 && ly >= 0 && lx < erosion_window_w_ && ly < erosion_window_h_) {
        const size_t index = static_cast<size_t>(ly) * erosion_window_w_ + lx;
        if (erosion_window_free_[index]) {
          return FREE_SPACE;
        }
        if (erosion_window_ring_[index]) {
          return LETHAL_OBSTACLE;
        }
      }
    }
    return getCost(mx, my);
  }

  /**
   * @brief Callback executed when a parameter change is detected
   * @param event ParameterEvent message
   */
  rcl_interfaces::msg::SetParametersResult
  dynamicParametersCallback(std::vector<rclcpp::Parameter> parameters);

  std::vector<geometry_msgs::msg::Point> transformed_footprint_;
  bool footprint_clearing_enabled_;
  /**
   * @brief Clear costmap layer info below the robot's footprint
   */
  void updateFootprint(
    double robot_x, double robot_y, double robot_yaw, double * min_x,
    double * min_y,
    double * max_x,
    double * max_y);

  std::string global_frame_;  ///< @brief The global frame for the costmap
  std::string map_frame_;  /// @brief frame that map is located in

  bool has_updated_data_{false};

  unsigned int x_{0};
  unsigned int y_{0};
  unsigned int width_{0};
  unsigned int height_{0};

  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Subscription<map_msgs::msg::OccupancyGridUpdate>::SharedPtr map_update_sub_;

  // Parameters
  std::string map_topic_;
  bool map_subscribe_transient_local_;
  bool subscribe_to_updates_;
  bool track_unknown_space_;
  bool use_maximum_;
  unsigned char lethal_threshold_;
  unsigned char unknown_cost_value_;
  bool trinary_costmap_;
  bool map_received_{false};
  bool map_received_in_update_bounds_{false};
  tf2::Duration transform_tolerance_;
  nav_msgs::msg::OccupancyGrid::SharedPtr map_buffer_;

  // Static map erosion. `erosion_radius_` and `erosion_ring_thickness_` are
  // load time only (the disc offsets are precomputed from them);
  // `erosion_enabled_` is a free runtime toggle so the way metadata can flip it
  // on every way transition.
  double erosion_radius_{0.0};
  double erosion_ring_thickness_{0.0};
  bool erosion_enabled_{false};
  bool erosion_active_{false};
  // Drivable region of the raw map, one bit per cell. This is the only full
  // map sized state the erosion keeps; the masks themselves are windowed.
  std::vector<bool> erosion_raw_free_;
  std::vector<std::pair<int, int>> erosion_free_offsets_;
  std::vector<std::pair<int, int>> erosion_ring_offsets_;
  int erosion_extent_{0};
  // Erosion masks for the currently cached window, in map cells.
  int erosion_window_x_{0};
  int erosion_window_y_{0};
  int erosion_window_w_{0};
  int erosion_window_h_{0};
  std::vector<bool> erosion_window_free_;
  std::vector<bool> erosion_window_ring_;
  // Dynamic parameters handler
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr dyn_params_handler_;
};

}  // namespace nav2_costmap_2d

#endif  // NAV2_COSTMAP_2D__STATIC_LAYER_HPP_
