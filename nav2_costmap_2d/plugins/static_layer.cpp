/*********************************************************************
 *
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2008, 2013, Willow Garage, Inc.
 *  Copyright (c) 2015, Fetch Robotics, Inc.
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

#include "nav2_costmap_2d/static_layer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "pluginlib/class_list_macros.hpp"
#include "tf2/convert.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "nav2_util/validate_messages.hpp"

PLUGINLIB_EXPORT_CLASS(nav2_costmap_2d::StaticLayer, nav2_costmap_2d::Layer)

using nav2_costmap_2d::NO_INFORMATION;
using nav2_costmap_2d::LETHAL_OBSTACLE;
using nav2_costmap_2d::FREE_SPACE;
using rcl_interfaces::msg::ParameterType;

namespace nav2_costmap_2d
{

StaticLayer::StaticLayer()
: map_buffer_(nullptr)
{
}

StaticLayer::~StaticLayer()
{
}

void
StaticLayer::onInitialize()
{
  global_frame_ = layered_costmap_->getGlobalFrameID();

  getParameters();

  rclcpp::QoS map_qos(10);  // initialize to default
  if (map_subscribe_transient_local_) {
    map_qos.transient_local();
    map_qos.reliable();
    map_qos.keep_last(1);
  }

  RCLCPP_INFO(
    logger_,
    "Subscribing to the map topic (%s) with %s durability",
    map_topic_.c_str(),
    map_subscribe_transient_local_ ? "transient local" : "volatile");

  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error{"Failed to lock node"};
  }

  map_sub_ = node->create_subscription<nav_msgs::msg::OccupancyGrid>(
    map_topic_, map_qos,
    std::bind(&StaticLayer::incomingMap, this, std::placeholders::_1));

  if (subscribe_to_updates_) {
    RCLCPP_INFO(logger_, "Subscribing to updates");
    map_update_sub_ = node->create_subscription<map_msgs::msg::OccupancyGridUpdate>(
      map_topic_ + "_updates",
      rclcpp::SystemDefaultsQoS(),
      std::bind(&StaticLayer::incomingUpdate, this, std::placeholders::_1));
  }
}

void
StaticLayer::activate()
{
  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error{"Failed to lock node"};
  }

  // Restore callback for dynamic parameters.
  dyn_params_handler_ = node->add_on_set_parameters_callback(
    std::bind(
      &StaticLayer::dynamicParametersCallback,
      this, std::placeholders::_1));

  // Always enable the static layer on activation to ensure map is shown.
  if (!enabled_) {
    RCLCPP_INFO(logger_, "Enabling static layer on activation");
    enabled_ = true;
    
    // Update the parameter to keep it consistent.
    auto param = rclcpp::Parameter(name_ + "." + "enabled", enabled_);
    node->set_parameter(param);
    
    // Mark data as updated to trigger map repaint
    x_ = y_ = 0;
    width_ = size_x_;
    height_ = size_y_;
    has_updated_data_ = true;
    current_ = false;
  }
}

void
StaticLayer::deactivate()
{
  auto node = node_.lock();
  if (dyn_params_handler_ && node) {
    node->remove_on_set_parameters_callback(dyn_params_handler_.get());
  }
  dyn_params_handler_.reset();
}

void
StaticLayer::reset()
{
  has_updated_data_ = true;
  current_ = false;
}

void
StaticLayer::getParameters()
{
  int temp_lethal_threshold = 0;
  double temp_tf_tol = 0.0;

  declareParameter("enabled", rclcpp::ParameterValue(true));
  declareParameter("subscribe_to_updates", rclcpp::ParameterValue(false));
  declareParameter("map_subscribe_transient_local", rclcpp::ParameterValue(true));
  declareParameter("transform_tolerance", rclcpp::ParameterValue(0.0));
  declareParameter("map_topic", rclcpp::ParameterValue("map"));
  declareParameter("footprint_clearing_enabled", rclcpp::ParameterValue(false));
  declareParameter("erosion_radius", rclcpp::ParameterValue(0.0));
  declareParameter("erosion_ring_thickness", rclcpp::ParameterValue(0.2));
  declareParameter("erosion_enabled", rclcpp::ParameterValue(false));

  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error{"Failed to lock node"};
  }

  node->get_parameter(name_ + "." + "enabled", enabled_);
  node->get_parameter(name_ + "." + "subscribe_to_updates", subscribe_to_updates_);
  node->get_parameter(name_ + "." + "footprint_clearing_enabled", footprint_clearing_enabled_);
  node->get_parameter(name_ + "." + "map_topic", map_topic_);
  node->get_parameter(name_ + "." + "erosion_radius", erosion_radius_);
  node->get_parameter(name_ + "." + "erosion_ring_thickness", erosion_ring_thickness_);
  node->get_parameter(name_ + "." + "erosion_enabled", erosion_enabled_);
  map_topic_ = joinWithParentNamespace(map_topic_);
  node->get_parameter(
    name_ + "." + "map_subscribe_transient_local",
    map_subscribe_transient_local_);
  node->get_parameter("track_unknown_space", track_unknown_space_);
  node->get_parameter("use_maximum", use_maximum_);
  node->get_parameter("lethal_cost_threshold", temp_lethal_threshold);
  node->get_parameter("unknown_cost_value", unknown_cost_value_);
  node->get_parameter("trinary_costmap", trinary_costmap_);
  node->get_parameter("transform_tolerance", temp_tf_tol);

  // Enforce bounds
  lethal_threshold_ = std::max(std::min(temp_lethal_threshold, 100), 0);
  map_received_ = false;
  map_received_in_update_bounds_ = false;

  transform_tolerance_ = tf2::durationFromSec(temp_tf_tol);

  // Add callback for dynamic parameters
  dyn_params_handler_ = node->add_on_set_parameters_callback(
    std::bind(
      &StaticLayer::dynamicParametersCallback,
      this, std::placeholders::_1));
}

void
StaticLayer::processMap(const nav_msgs::msg::OccupancyGrid & new_map)
{
  RCLCPP_DEBUG(logger_, "StaticLayer: Process map");

  unsigned int size_x = new_map.info.width;
  unsigned int size_y = new_map.info.height;

  RCLCPP_DEBUG(
    logger_,
    "StaticLayer: Received a %d X %d map at %f m/pix", size_x, size_y,
    new_map.info.resolution);

  // resize costmap if size, resolution or origin do not match
  Costmap2D * master = layered_costmap_->getCostmap();
  if (!layered_costmap_->isRolling() && (master->getSizeInCellsX() != size_x ||
    master->getSizeInCellsY() != size_y ||
    master->getResolution() != new_map.info.resolution ||
    master->getOriginX() != new_map.info.origin.position.x ||
    master->getOriginY() != new_map.info.origin.position.y ||
    !layered_costmap_->isSizeLocked()))
  {
    // Update the size of the layered costmap (and all layers, including this one)
    RCLCPP_INFO(
      logger_,
      "StaticLayer: Resizing costmap to %d X %d at %f m/pix", size_x, size_y,
      new_map.info.resolution);
    layered_costmap_->resizeMap(
      size_x, size_y, new_map.info.resolution,
      new_map.info.origin.position.x,
      new_map.info.origin.position.y,
      true);
  } else if (size_x_ != size_x || size_y_ != size_y ||  // NOLINT
    resolution_ != new_map.info.resolution ||
    origin_x_ != new_map.info.origin.position.x ||
    origin_y_ != new_map.info.origin.position.y)
  {
    // only update the size of the costmap stored locally in this layer
    RCLCPP_INFO(
      logger_,
      "StaticLayer: Resizing static layer to %d X %d at %f m/pix", size_x, size_y,
      new_map.info.resolution);
    resizeMap(
      size_x, size_y, new_map.info.resolution,
      new_map.info.origin.position.x, new_map.info.origin.position.y);
  }

  unsigned int index = 0;

  // we have a new map, update full size of map
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());

  // initialize the costmap with static data
  for (unsigned int i = 0; i < size_y; ++i) {
    for (unsigned int j = 0; j < size_x; ++j) {
      unsigned char value = new_map.data[index];
      costmap_[index] = interpretValue(value);
      ++index;
    }
  }

  erosion_window_w_ = 0;
  erosion_window_h_ = 0;
  buildRawFreeMask(new_map);

  map_frame_ = new_map.header.frame_id;

  x_ = y_ = 0;
  width_ = size_x_;
  height_ = size_y_;
  has_updated_data_ = true;

  current_ = true;
}

void
StaticLayer::buildRawFreeMask(const nav_msgs::msg::OccupancyGrid & new_map)
{
  erosion_raw_free_.clear();
  erosion_raw_free_.shrink_to_fit();
  erosion_window_free_.clear();
  erosion_window_ring_.clear();
  erosion_window_w_ = 0;
  erosion_window_h_ = 0;
  erosion_active_ = false;

  if (erosion_radius_ <= 0.0) {
    return;
  }

  const size_t num_cells = static_cast<size_t>(new_map.info.width) * new_map.info.height;
  if (num_cells == 0 || new_map.data.size() != num_cells) {
    RCLCPP_ERROR(logger_, "StaticLayer: malformed map, cannot erode the static map.");
    return;
  }
  if (new_map.info.resolution <= 0.0) {
    RCLCPP_ERROR(
      logger_, "StaticLayer: map resolution is %f, cannot erode the static map. "
      "Erosion disabled.", new_map.info.resolution);
    return;
  }

  erosion_raw_free_.assign(num_cells, false);
  size_t free_cells = 0;
  for (size_t i = 0; i < num_cells; ++i) {
    if (isRawFree(static_cast<unsigned char>(new_map.data[i]))) {
      erosion_raw_free_[i] = true;
      ++free_cells;
    }
  }

  computeErosionOffsets();
  erosion_active_ = erosion_enabled_ && !erosion_free_offsets_.empty();

  RCLCPP_INFO(
    logger_,
    "StaticLayer: static map erosion ready over %zu drivable cells: grows the "
    "drivable region by %.2fm and redraws a %.2fm boundary, computed per "
    "costmap window. Currently %s.",
    free_cells, erosion_radius_, erosion_ring_thickness_,
    erosion_active_ ? "enabled" : "disabled");
}

void
StaticLayer::computeErosionOffsets()
{
  erosion_free_offsets_.clear();
  erosion_ring_offsets_.clear();
  erosion_extent_ = 0;

  if (erosion_radius_ <= 0.0 || resolution_ <= 0.0) {
    return;
  }

  const double free_radius_cells = erosion_radius_ / resolution_;
  const double ring_radius_cells = (erosion_radius_ + erosion_ring_thickness_) / resolution_;
  // Radii are compared with a RELATIVE tolerance: radius / resolution is not
  // exact in binary floating point, and info.resolution is a float32, so
  // 1.0m / 0.1 yields 9.99999985 cells rather than 10. Without this the
  // outermost cells of the disc are silently dropped (~0.15% of a real
  // segmapping). The tolerance must scale with the radius, an absolute one is
  // large enough at 3 cells and too small at 10.
  constexpr double kRadiusRelativeTolerance = 1e-6;
  const double free_radius_sq =
    free_radius_cells * free_radius_cells * (1.0 + kRadiusRelativeTolerance);
  const double ring_radius_sq =
    ring_radius_cells * ring_radius_cells * (1.0 + kRadiusRelativeTolerance);
  erosion_extent_ = static_cast<int>(std::ceil(ring_radius_cells));

  for (int dy = -erosion_extent_; dy <= erosion_extent_; ++dy) {
    for (int dx = -erosion_extent_; dx <= erosion_extent_; ++dx) {
      const double dist_sq = static_cast<double>(dx) * dx + static_cast<double>(dy) * dy;
      if (dist_sq <= free_radius_sq) {
        erosion_free_offsets_.emplace_back(dx, dy);
      } else if (dist_sq <= ring_radius_sq) {
        erosion_ring_offsets_.emplace_back(dx, dy);
      }
    }
  }
}

void
StaticLayer::updateErosionWindow(int x0, int y0, int x1, int y1)
{
  if (!erosion_active_) {
    return;
  }

  // Grow the requested rectangle by the ring radius: a drivable cell just
  // outside it can still push the boundary into it.
  const int map_w = static_cast<int>(size_x_);
  const int map_h = static_cast<int>(size_y_);
  const int wx0 = std::max(0, x0 - erosion_extent_);
  const int wy0 = std::max(0, y0 - erosion_extent_);
  const int wx1 = std::min(map_w - 1, x1 + erosion_extent_);
  const int wy1 = std::min(map_h - 1, y1 + erosion_extent_);
  if (wx1 < wx0 || wy1 < wy0) {
    erosion_window_w_ = 0;
    erosion_window_h_ = 0;
    return;
  }

  const int width = wx1 - wx0 + 1;
  const int height = wy1 - wy0 + 1;
  if (wx0 == erosion_window_x_ && wy0 == erosion_window_y_ &&
    width == erosion_window_w_ && height == erosion_window_h_)
  {
    return;  // the robot has not moved far enough to change the window
  }

  erosion_window_x_ = wx0;
  erosion_window_y_ = wy0;
  erosion_window_w_ = width;
  erosion_window_h_ = height;
  const size_t window_cells = static_cast<size_t>(width) * height;
  erosion_window_free_.assign(window_cells, false);
  erosion_window_ring_.assign(window_cells, false);

  auto raw_free_at = [this, map_w](int mx, int my) {
      return erosion_raw_free_[static_cast<size_t>(my) * map_w + mx];
    };

  // Seed the growth with the drivable cells that have a non-drivable
  // 8-neighbour: only the border of the region can push the boundary outwards.
  // Neighbours are read from the full map mask, so the window edges are not a
  // special case.
  std::vector<std::pair<int, int>> seeds;
  for (int my = wy0; my <= wy1; ++my) {
    for (int mx = wx0; mx <= wx1; ++mx) {
      if (!raw_free_at(mx, my)) {
        continue;
      }
      erosion_window_free_[static_cast<size_t>(my - wy0) * width + (mx - wx0)] = true;
      bool on_border = false;
      for (int dy = -1; dy <= 1 && !on_border; ++dy) {
        for (int dx = -1; dx <= 1 && !on_border; ++dx) {
          if (dx == 0 && dy == 0) {
            continue;
          }
          const int nx = mx + dx;
          const int ny = my + dy;
          if (nx < 0 || ny < 0 || nx >= map_w || ny >= map_h) {
            continue;
          }
          if (!raw_free_at(nx, ny)) {
            on_border = true;
          }
        }
      }
      if (on_border) {
        seeds.emplace_back(mx, my);
      }
    }
  }

  auto stamp = [&](const std::vector<std::pair<int, int>> & offsets, bool ring) {
      for (const auto & seed : seeds) {
        for (const auto & offset : offsets) {
          const int lx = seed.first + offset.first - wx0;
          const int ly = seed.second + offset.second - wy0;
          if (lx < 0 || ly < 0 || lx >= width || ly >= height) {
            continue;
          }
          const size_t index = static_cast<size_t>(ly) * width + lx;
          if (!ring) {
            erosion_window_free_[index] = true;
          } else if (!erosion_window_free_[index]) {
            // Tested against the finished free mask, so a cell is never both
            // drivable and boundary and no second pass is needed.
            erosion_window_ring_[index] = true;
          }
        }
      }
    };

  stamp(erosion_free_offsets_, false);
  stamp(erosion_ring_offsets_, true);
}

void
StaticLayer::matchSize()
{
  // If we are using rolling costmap, the static map size is
  //   unrelated to the size of the layered costmap
  if (!layered_costmap_->isRolling()) {
    Costmap2D * master = layered_costmap_->getCostmap();
    resizeMap(
      master->getSizeInCellsX(), master->getSizeInCellsY(), master->getResolution(),
      master->getOriginX(), master->getOriginY());
  }
}

unsigned char
StaticLayer::interpretValue(unsigned char value)
{
  // check if the static value is above the unknown or lethal thresholds
  if (track_unknown_space_ && value == unknown_cost_value_) {
    return NO_INFORMATION;
  } else if (!track_unknown_space_ && value == unknown_cost_value_) {
    return FREE_SPACE;
  } else if (value >= lethal_threshold_) {
    return LETHAL_OBSTACLE;
  } else if (trinary_costmap_) {
    return FREE_SPACE;
  }

  double scale = static_cast<double>(value) / lethal_threshold_;
  return scale * LETHAL_OBSTACLE;
}

void
StaticLayer::incomingMap(const nav_msgs::msg::OccupancyGrid::SharedPtr new_map)
{
  if (!nav2_util::validateMsg(*new_map)) {
    RCLCPP_ERROR(logger_, "Received map message is malformed. Rejecting.");
    return;
  }
  if (!map_received_) {
    processMap(*new_map);
    map_received_ = true;
    return;
  }
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  map_buffer_ = new_map;
}

void
StaticLayer::incomingUpdate(map_msgs::msg::OccupancyGridUpdate::ConstSharedPtr update)
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  if (update->y < static_cast<int32_t>(y_) ||
    y_ + height_ < update->y + update->height ||
    update->x < static_cast<int32_t>(x_) ||
    x_ + width_ < update->x + update->width)
  {
    RCLCPP_WARN(
      logger_,
      "StaticLayer: Map update ignored. Exceeds bounds of static layer.\n"
      "Static layer origin: %d, %d   bounds: %d X %d\n"
      "Update origin: %d, %d   bounds: %d X %d",
      x_, y_, width_, height_, update->x, update->y, update->width,
      update->height);
    return;
  }

  if (update->header.frame_id != map_frame_) {
    RCLCPP_WARN(
      logger_,
      "StaticLayer: Map update ignored. Current map is in frame %s "
      "but update was in frame %s",
      map_frame_.c_str(), update->header.frame_id.c_str());
    return;
  }

  const bool track_erosion = !erosion_raw_free_.empty();

  unsigned int di = 0;
  for (unsigned int y = 0; y < update->height; y++) {
    unsigned int index_base = (update->y + y) * size_x_;
    for (unsigned int x = 0; x < update->width; x++) {
      unsigned int index = index_base + x + update->x;
      const unsigned char raw = static_cast<unsigned char>(update->data[di]);
      costmap_[index] = interpretValue(update->data[di++]);
      if (track_erosion) {
        // Keep the drivable region in step with partial updates, otherwise the
        // erosion would keep growing from cells the map no longer reports.
        erosion_raw_free_[index] = isRawFree(raw);
      }
    }
  }

  if (track_erosion) {
    // Drop the cached window so it is rebuilt from the updated region.
    erosion_window_w_ = 0;
    erosion_window_h_ = 0;
  }

  has_updated_data_ = true;
}


void
StaticLayer::updateBounds(
  double robot_x, double robot_y, double robot_yaw, double * min_x,
  double * min_y,
  double * max_x,
  double * max_y)
{
  if (!map_received_) {
    map_received_in_update_bounds_ = false;
    return;
  }
  map_received_in_update_bounds_ = true;

  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());

  // If there is a new available map, load it.
  if (map_buffer_) {
    processMap(*map_buffer_);
    map_buffer_ = nullptr;
  }

  if (!layered_costmap_->isRolling() ) {
    if (!(has_updated_data_ || has_extra_bounds_)) {
      return;
    }
  }

  useExtraBounds(min_x, min_y, max_x, max_y);

  double wx, wy;

  mapToWorld(x_, y_, wx, wy);
  *min_x = std::min(wx, *min_x);
  *min_y = std::min(wy, *min_y);

  mapToWorld(x_ + width_, y_ + height_, wx, wy);
  *max_x = std::max(wx, *max_x);
  *max_y = std::max(wy, *max_y);

  has_updated_data_ = false;

  updateFootprint(robot_x, robot_y, robot_yaw, min_x, min_y, max_x, max_y);
}

void
StaticLayer::updateFootprint(
  double robot_x, double robot_y, double robot_yaw,
  double * min_x, double * min_y,
  double * max_x,
  double * max_y)
{
  if (!footprint_clearing_enabled_) {return;}

  transformFootprint(robot_x, robot_y, robot_yaw, getFootprint(), transformed_footprint_);

  for (unsigned int i = 0; i < transformed_footprint_.size(); i++) {
    touch(transformed_footprint_[i].x, transformed_footprint_[i].y, min_x, min_y, max_x, max_y);
  }
}

void
StaticLayer::updateCosts(
  nav2_costmap_2d::Costmap2D & master_grid,
  int min_i, int min_j, int max_i, int max_j)
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  if (!enabled_) {
    return;
  }
  if (!map_received_in_update_bounds_) {
    static int count = 0;
    // throttle warning down to only 1/10 message rate
    if (++count == 10) {
      RCLCPP_WARN(logger_, "Can't update static costmap layer, no map received");
      count = 0;
    }
    return;
  }

  if (footprint_clearing_enabled_) {
    setConvexPolygonCost(transformed_footprint_, nav2_costmap_2d::FREE_SPACE);
  }

  if (!layered_costmap_->isRolling()) {
    // if not rolling, the layered costmap (master_grid) has same coordinates as this layer
    if (erosion_active_) {
      updateErosionWindow(min_i, min_j, max_i - 1, max_j - 1);
      // Same as updateWithTrueOverwrite/updateWithMax, but reading through the
      // erosion masks instead of straight out of costmap_.
      for (int j = min_j; j < max_j; ++j) {
        for (int i = min_i; i < max_i; ++i) {
          const unsigned char cost = getStaticCost(i, j);
          if (!use_maximum_) {
            master_grid.setCost(i, j, cost);
          } else {
            master_grid.setCost(i, j, std::max(cost, master_grid.getCost(i, j)));
          }
        }
      }
    } else if (!use_maximum_) {
      updateWithTrueOverwrite(master_grid, min_i, min_j, max_i, max_j);
    } else {
      updateWithMax(master_grid, min_i, min_j, max_i, max_j);
    }
  } else {
    // If rolling window, the master_grid is unlikely to have same coordinates as this layer
    unsigned int mx, my;
    double wx, wy;
    // Might even be in a different frame
    geometry_msgs::msg::TransformStamped transform;
    try {
      transform = tf_->lookupTransform(
        map_frame_, global_frame_, tf2::TimePointZero,
        transform_tolerance_);
    } catch (tf2::TransformException & ex) {
      RCLCPP_ERROR(logger_, "StaticLayer: %s", ex.what());
      return;
    }
    // Copy map data given proper transformations
    tf2::Transform tf2_transform;
    tf2::fromMsg(transform.transform, tf2_transform);

    if (erosion_active_) {
      // Erode only the slice of the map this window reads. The transform is
      // rigid, so the axis aligned box of the four transformed corners covers
      // every cell the loop below can reach.
      int rx0 = std::numeric_limits<int>::max();
      int ry0 = std::numeric_limits<int>::max();
      int rx1 = std::numeric_limits<int>::min();
      int ry1 = std::numeric_limits<int>::min();
      const int corners_i[4] = {min_i, max_i - 1, min_i, max_i - 1};
      const int corners_j[4] = {min_j, min_j, max_j - 1, max_j - 1};
      for (int corner = 0; corner < 4; ++corner) {
        layered_costmap_->getCostmap()->mapToWorld(corners_i[corner], corners_j[corner], wx, wy);
        tf2::Vector3 corner_point(wx, wy, 0);
        corner_point = tf2_transform * corner_point;
        // Not worldToMap(), which fails outside the map and would lose the bound.
        const int cell_x =
          static_cast<int>(std::floor((corner_point.x() - origin_x_) / resolution_));
        const int cell_y =
          static_cast<int>(std::floor((corner_point.y() - origin_y_) / resolution_));
        rx0 = std::min(rx0, cell_x);
        ry0 = std::min(ry0, cell_y);
        rx1 = std::max(rx1, cell_x);
        ry1 = std::max(ry1, cell_y);
      }
      updateErosionWindow(rx0, ry0, rx1, ry1);
    }

    for (int i = min_i; i < max_i; ++i) {
      for (int j = min_j; j < max_j; ++j) {
        // Convert master_grid coordinates (i,j) into global_frame_(wx,wy) coordinates
        layered_costmap_->getCostmap()->mapToWorld(i, j, wx, wy);
        // Transform from global_frame_ to map_frame_
        tf2::Vector3 p(wx, wy, 0);
        p = tf2_transform * p;
        // Set master_grid with cell from map
        if (worldToMap(p.x(), p.y(), mx, my)) {
          const unsigned char cost = getStaticCost(mx, my);
          if (!use_maximum_) {
            master_grid.setCost(i, j, cost);
          } else {
            master_grid.setCost(i, j, std::max(cost, master_grid.getCost(i, j)));
          }
        }
      }
    }
  }
  current_ = true;
}

/**
  * @brief Callback executed when a parameter change is detected
  * @param event ParameterEvent message
  */
rcl_interfaces::msg::SetParametersResult
StaticLayer::dynamicParametersCallback(
  std::vector<rclcpp::Parameter> parameters)
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  rcl_interfaces::msg::SetParametersResult result;

  for (auto parameter : parameters) {
    const auto & param_type = parameter.get_type();
    const auto & param_name = parameter.get_name();

    if (param_name == name_ + "." + "erosion_radius" ||
      param_name == name_ + "." + "erosion_ring_thickness")
    {
      // Actually reject these, rather than only warning: the masks are built
      // once from the whole map, so a new value would not be applied, and a
      // parameter that reads back as changed while the layer keeps using the
      // old one is a trap when debugging on a robot. Use erosion_enabled to
      // turn the erosion on and off at runtime.
      RCLCPP_WARN(
        logger_, "%s is not a dynamic parameter and cannot be changed while "
        "running. Rejecting parameter update.", param_name.c_str());
      result.successful = false;
      result.reason = param_name + " is a load time parameter of the static layer";
      return result;
    } else if (param_name == name_ + "." + "map_subscribe_transient_local" || // NOLINT
      param_name == name_ + "." + "map_topic" ||
      param_name == name_ + "." + "subscribe_to_updates")
    {
      RCLCPP_WARN(
        logger_, "%s is not a dynamic parameter "
        "cannot be changed while running. Rejecting parameter update.", param_name.c_str());
    } else if (param_type == ParameterType::PARAMETER_DOUBLE) {
      if (param_name == name_ + "." + "transform_tolerance") {
        transform_tolerance_ = tf2::durationFromSec(parameter.as_double());
      }
    } else if (param_type == ParameterType::PARAMETER_BOOL) {
      if (param_name == name_ + "." + "enabled" && enabled_ != parameter.as_bool()) {
        enabled_ = parameter.as_bool();

        x_ = y_ = 0;
        width_ = size_x_;
        height_ = size_y_;
        has_updated_data_ = true;
        current_ = false;
      } else if (param_name == name_ + "." + "footprint_clearing_enabled") {
        footprint_clearing_enabled_ = parameter.as_bool();
      } else if (param_name == name_ + "." + "erosion_enabled" && // NOLINT
        erosion_enabled_ != parameter.as_bool())
      {
        erosion_enabled_ = parameter.as_bool();
        erosion_active_ = erosion_enabled_ && !erosion_raw_free_.empty() &&
          !erosion_free_offsets_.empty();
        erosion_window_w_ = 0;
        erosion_window_h_ = 0;
        if (erosion_enabled_ && erosion_radius_ <= 0.0) {
          RCLCPP_WARN(
            logger_,
            "StaticLayer: erosion_enabled was set but erosion_radius is 0, so "
            "the erosion does nothing. Set erosion_radius > 0 at startup.");
        } else if (erosion_enabled_ && erosion_raw_free_.empty()) {
          RCLCPP_INFO(
            logger_,
            "StaticLayer: erosion_enabled was set before the map arrived. The "
            "erosion will start as soon as the map is received.");
        }

        // Redraw the whole layer so the change is applied everywhere at once.
        x_ = y_ = 0;
        width_ = size_x_;
        height_ = size_y_;
        has_updated_data_ = true;
        current_ = false;
      }
    }
  }
  result.successful = true;
  return result;
}

}  // namespace nav2_costmap_2d
