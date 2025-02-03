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
 *********************************************************************/
#include "nav2_costmap_2d/segmentation_buffer.hpp"

#include <algorithm>
#include <chrono>
#include <list>
#include <string>
#include <vector>

#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "tf2/convert.h"
#include "rclcpp/rclcpp.hpp"
using namespace std::chrono_literals;

namespace nav2_costmap_2d {
SegmentationBuffer::SegmentationBuffer(const nav2_util::LifecycleNode::WeakPtr& parent,
                                       std::string buffer_source, std::vector<std::string> class_types, std::unordered_map<std::string, CostHeuristicParams> class_names_cost_map, double observation_keep_time,
                                       double expected_update_rate, double max_lookahead_distance,
                                       double min_lookahead_distance, tf2_ros::Buffer& tf2_buffer,
                                       std::string global_frame, std::string sensor_frame,
                                       tf2::Duration tf_tolerance, double costmap_resolution, double tile_map_decay_time, bool visualize_tile_map)
  : tf2_buffer_(tf2_buffer)
  , class_types_(class_types)
  , class_names_cost_map_(class_names_cost_map)
  , observation_keep_time_(rclcpp::Duration::from_seconds(observation_keep_time))
  , expected_update_rate_(rclcpp::Duration::from_seconds(expected_update_rate))
  , global_frame_(global_frame)
  , sensor_frame_(sensor_frame)
  , buffer_source_(buffer_source)
  , sq_max_lookahead_distance_(std::pow(max_lookahead_distance, 2))
  , sq_min_lookahead_distance_(std::pow(min_lookahead_distance, 2))
  , tf_tolerance_(tf_tolerance)
{
  auto node = parent.lock();
  clock_ = node->get_clock();
  logger_ = node->get_logger();
  last_updated_ = node->now();
  temporal_tile_map_ = std::make_shared<SegmentationTileMap>(costmap_resolution, tile_map_decay_time);
  visualize_tile_map_ = visualize_tile_map;
  if(visualize_tile_map_)
  {
    tile_map_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>(buffer_source + "/tile_map",1);
  }
}

SegmentationBuffer::~SegmentationBuffer() {}

void SegmentationBuffer::createSegmentationCostMultimap(const vision_msgs::msg::LabelInfo& label_info)
{
  std::unordered_map<std::string, uint8_t> class_to_id_map;
  for (const auto& semantic_class : label_info.class_map)
    {
      class_to_id_map[semantic_class.class_name] = semantic_class.class_id;
    }
    segmentation_cost_multimap_ = std::make_shared<SegmentationCostMultimap>(class_to_id_map, class_names_cost_map_);
}

void SegmentationBuffer::bufferSegmentation(
  const sensor_msgs::msg::PointCloud2& cloud,
  const sensor_msgs::msg::Image& segmentation,
  const sensor_msgs::msg::Image& confidence)
{
  geometry_msgs::msg::PointStamped global_origin;
  // check whether the origin frame has been set explicitly
  // or whether we should get it from the cloud
  std::string origin_frame = sensor_frame_ == "" ? cloud.header.frame_id : sensor_frame_;

  try
  {
    // given these segmentations come from sensors...
    // we'll need to store the origin pt of the sensor
    geometry_msgs::msg::PointStamped local_origin;
    local_origin.header.stamp = cloud.header.stamp;
    local_origin.header.frame_id = origin_frame;
    local_origin.point.x = 0;
    local_origin.point.y = 0;
    local_origin.point.z = 0;
    tf2_buffer_.transform(local_origin, global_origin, global_frame_, tf_tolerance_);

    sensor_msgs::msg::PointCloud2 global_frame_cloud;

    // transform the point cloud
    tf2_buffer_.transform(cloud, global_frame_cloud, global_frame_, tf_tolerance_);
    global_frame_cloud.header.stamp = cloud.header.stamp;

    sensor_msgs::PointCloud2ConstIterator<float> iter_x_global(global_frame_cloud, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y_global(global_frame_cloud, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z_global(global_frame_cloud, "z");
    std::unordered_map<TileIndex, int> best_observations_idxs;
    double cloud_time_seconds = rclcpp::Time(cloud.header.stamp.sec, cloud.header.stamp.nanosec).seconds();

    // copy over the points that are within our segmentation range
    for (size_t v = 0; v < segmentation.height; v++)
    {
      for (size_t u = 0; u < segmentation.width; u++)
      {
        int pixel_idx = v * segmentation.width + u;
        // remove invalid points
        if (!std::isfinite(*(iter_z_global)))
        {
          ++iter_x_global;
          ++iter_y_global;
          ++iter_z_global;
          continue;
        }
        double sq_dist =
          std::pow(*(iter_x_global) - global_origin.point.x, 2) +
          std::pow(*(iter_y_global) - global_origin.point.y, 2) +
          std::pow(*(iter_z_global) - global_origin.point.z, 2);
        if (sq_dist >= sq_max_lookahead_distance_ || sq_dist <= sq_min_lookahead_distance_)
        {
          ++iter_x_global;
          ++iter_y_global;
          ++iter_z_global;
          continue;
        }

        TileIndex costmap_index = temporal_tile_map_->worldToIndex(*iter_x_global, *iter_y_global);

        // Find the observation with the greater confidence for each tile
        auto it = best_observations_idxs.find(costmap_index);
        // if an observation already exists in the index, compare its confidence to the current
        // one and if its greater store this element as the new best confidence index for the tile
        if (it != best_observations_idxs.end()) {
          if(confidence.data[pixel_idx] > confidence.data[best_observations_idxs[costmap_index]])
          {
            best_observations_idxs[costmap_index] = pixel_idx;
          }
        }
        // if this is the first observation for the tile index just store its confidence as the best one
        else
        {
          best_observations_idxs[costmap_index] = pixel_idx;
        }
        ++iter_x_global;
        ++iter_y_global;
        ++iter_z_global;
      }
    }

    // emplace the best observations in the mask into the tile map
    temporal_tile_map_->lock();
    temporal_tile_map_->purgeOldObservations(cloud_time_seconds);
    for (auto& idx : best_observations_idxs)
    {
      int img_idx_for_best_obs = idx.second;
      TileIndex costmap_index = idx.first;
      TileObservation best_obs{segmentation.data[img_idx_for_best_obs], static_cast<float>(confidence.data[img_idx_for_best_obs]), cloud_time_seconds};
      temporal_tile_map_->pushObservation(best_obs, costmap_index);
    }
    temporal_tile_map_->unlock();

    if(visualize_tile_map_)
    {
      sensor_msgs::msg::PointCloud2 tile_map_cloud = visualizeTemporalTileMap(*temporal_tile_map_);
      tile_map_pub_->publish(tile_map_cloud);
    }

  } catch (tf2::TransformException& ex)
  {
    RCLCPP_ERROR(logger_,
                 "TF Exception that should never happen for sensor frame: %s, cloud frame: %s, %s",
                 sensor_frame_.c_str(), cloud.header.frame_id.c_str(), ex.what());
    return;
  }

  // if the update was successful, we want to update the last updated time
  last_updated_ = clock_->now();
}


std::unordered_map<std::string, CostHeuristicParams> SegmentationBuffer::getClassMap()
{
  return class_names_cost_map_;
}


void SegmentationBuffer::updateClassMap(std::string new_class, CostHeuristicParams new_cost)
{
  segmentation_cost_multimap_->updateCostByName(new_class, new_cost);
}

bool SegmentationBuffer::isCurrent() const
{
  if (expected_update_rate_ == rclcpp::Duration(0.0s))
  {
    return true;
  }

  bool current = (clock_->now() - last_updated_) <= expected_update_rate_;
  if (!current)
  {
    RCLCPP_WARN(logger_,
                "The %s segmentation buffer has not been updated for %.2f seconds, "
                "and it should be updated every %.2f seconds.",
                buffer_source_.c_str(), (clock_->now() - last_updated_).seconds(),
                expected_update_rate_.seconds());
  }
  return current;
}

void SegmentationBuffer::resetLastUpdated() { last_updated_ = clock_->now(); }
}  // namespace nav2_costmap_2d
