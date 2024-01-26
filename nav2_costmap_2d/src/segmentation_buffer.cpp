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
using namespace std::chrono_literals;

namespace nav2_costmap_2d {
SegmentationBuffer::SegmentationBuffer(const nav2_util::LifecycleNode::WeakPtr& parent,
                                       std::string buffer_source, std::vector<std::string> class_types, std::unordered_map<std::string, uint8_t> class_names_cost_map, double observation_keep_time,
                                       double expected_update_rate, double max_lookahead_distance,
                                       double min_lookahead_distance, tf2_ros::Buffer& tf2_buffer,
                                       std::string global_frame, std::string sensor_frame,
                                       tf2::Duration tf_tolerance, double costmap_resolution)
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
  temporal_tile_map_ = SegmentationTileMap(costmap_resolution, observation_keep_time);
}

SegmentationBuffer::~SegmentationBuffer() {}

void SegmentationBuffer::createClassIdCostMap(const vision_msgs::msg::LabelInfo& label_info)
{
  for (const auto& semantic_class : label_info.class_map)
    {
      class_ids_cost_map_[semantic_class.class_id] = class_names_cost_map_[semantic_class.class_name];
    }
}

void SegmentationBuffer::bufferSegmentation(
  const sensor_msgs::msg::PointCloud2& cloud,
  const sensor_msgs::msg::Image& segmentation,
  const sensor_msgs::msg::Image& confidence)
{
  geometry_msgs::msg::PointStamped global_origin;

  // create a new segmentation on the list to be populated
  segmentation_list_.push_front(Segmentation());

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
    tf2::convert(global_origin.point, segmentation_list_.front().origin_);

    sensor_msgs::msg::PointCloud2 global_frame_cloud;

    // transform the point cloud
    tf2_buffer_.transform(cloud, global_frame_cloud, global_frame_, tf_tolerance_);
    global_frame_cloud.header.stamp = cloud.header.stamp;

    // now we need to remove segmentations from the cloud that are below
    // or above our height thresholds
    sensor_msgs::msg::PointCloud2& segmentation_cloud = *(segmentation_list_.front().cloud_);
    segmentation_cloud.height = global_frame_cloud.height;
    segmentation_cloud.width = global_frame_cloud.width;
    segmentation_cloud.fields = global_frame_cloud.fields;
    segmentation_cloud.is_bigendian = global_frame_cloud.is_bigendian;
    segmentation_cloud.point_step = global_frame_cloud.point_step;
    segmentation_cloud.row_step = global_frame_cloud.row_step;
    segmentation_cloud.is_dense = global_frame_cloud.is_dense;

    unsigned int cloud_size = global_frame_cloud.height * global_frame_cloud.width;
    sensor_msgs::PointCloud2Modifier modifier(segmentation_cloud);

    segmentation_cloud.point_step =
      addPointField(segmentation_cloud, "class", 1, sensor_msgs::msg::PointField::INT8,
                    segmentation_cloud.point_step);
    segmentation_cloud.point_step =
      addPointField(segmentation_cloud, "confidence", 1, sensor_msgs::msg::PointField::INT8,
                    segmentation_cloud.point_step);
    modifier.resize(cloud_size);
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_class_obs(segmentation_cloud, "class");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_confidence_obs(segmentation_cloud, "confidence");
    sensor_msgs::PointCloud2Iterator<float> iter_x_obs(segmentation_cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y_obs(segmentation_cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z_obs(segmentation_cloud, "z");

    sensor_msgs::PointCloud2ConstIterator<float> iter_x_global(global_frame_cloud, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y_global(global_frame_cloud, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z_global(global_frame_cloud, "z");
    unsigned int point_count = 0;
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
          std::pow(*(iter_x_global) - segmentation_list_.front().origin_.x, 2) +
          std::pow(*(iter_y_global) - segmentation_list_.front().origin_.y, 2) +
          std::pow(*(iter_z_global) - segmentation_list_.front().origin_.z, 2);
        if (sq_dist >= sq_max_lookahead_distance_ || sq_dist <= sq_min_lookahead_distance_)
        {
          ++iter_x_global;
          ++iter_y_global;
          ++iter_z_global;
          continue;
        }

        
        *(iter_class_obs) = segmentation.data[pixel_idx];
        *(iter_confidence_obs) = confidence.data[pixel_idx];
        *(iter_x_obs) = *(iter_x_global);
        *(iter_y_obs) = *(iter_y_global);
        *(iter_z_obs) = *(iter_z_global);
        point_count++;

        TileIndex costmap_index = temporal_tile_map_.worldToIndex(*iter_x_global, *iter_y_global);

        // Update best observation for each TileIndex
        auto it = best_observations_idxs.find(costmap_index);
        if (it != best_observations_idxs.end()) {
          best_observations_idxs[costmap_index] = pixel_idx;
        }
        else
        {
          if(confidence.data[pixel_idx] > confidence.data[best_observations_idxs[costmap_index]])
          {
            best_observations_idxs[costmap_index] = pixel_idx;
          }
        }
        ++iter_x_global;
        ++iter_y_global;
        ++iter_z_global;
        ++iter_x_obs;
        ++iter_y_obs;
        ++iter_z_obs;
        ++iter_class_obs;
        ++iter_confidence_obs;
      }
      // std::cout << "pushing " << best_observations_idxs.size() << " observations to tile map\n";
      for (auto& idx : best_observations_idxs)
      {
        int img_idx_for_best_obs = idx.second;
        TileIndex costmap_index = idx.first;
        TileObservation best_obs{segmentation.data[img_idx_for_best_obs], getCostForClassId(segmentation.data[img_idx_for_best_obs]), static_cast<float>(confidence.data[img_idx_for_best_obs]), cloud_time_seconds};
        temporal_tile_map_.pushObservation(best_obs, costmap_index);
      }
    }

    // resize the cloud for the number of legal points
    modifier.resize(point_count);
    segmentation_cloud.header.stamp = cloud.header.stamp;
    segmentation_cloud.header.frame_id = global_frame_cloud.header.frame_id;

    segmentation_list_.front().class_map_ = class_ids_cost_map_;
  } catch (tf2::TransformException& ex)
  {
    // if an exception occurs, we need to remove the empty segmentation from the list
    segmentation_list_.pop_front();
    RCLCPP_ERROR(logger_,
                 "TF Exception that should never happen for sensor frame: %s, cloud frame: %s, %s",
                 sensor_frame_.c_str(), cloud.header.frame_id.c_str(), ex.what());
    return;
  }

  // if the update was successful, we want to update the last updated time
  last_updated_ = clock_->now();

  // we'll also remove any stale segmentations from the list
  purgeStaleSegmentations();
}

// returns a copy of the segmentations
void SegmentationBuffer::getSegmentations(std::vector<Segmentation>& segmentations)
{
  // first... let's make sure that we don't have any stale segmentations
  purgeStaleSegmentations();

  // now we'll just copy the segmentations for the caller
  std::list<Segmentation>::iterator obs_it;
  for (obs_it = segmentation_list_.begin(); obs_it != segmentation_list_.end(); ++obs_it)
  {
    segmentations.push_back(*obs_it);
  }
  segmentation_list_.clear();
}

std::unordered_map<std::string, uint8_t> SegmentationBuffer::getClassMap()
{
  return class_names_cost_map_;
}


void SegmentationBuffer::purgeStaleSegmentations()
{
  if (!segmentation_list_.empty())
  {
    std::list<Segmentation>::iterator obs_it = segmentation_list_.begin();
    // if we're keeping segmentations for no time... then we'll only keep one segmentation
    if (observation_keep_time_ == rclcpp::Duration(0.0s))
    {
      segmentation_list_.erase(++obs_it, segmentation_list_.end());
      return;
    }

    // otherwise... we'll have to loop through the segmentations to see which ones are stale
    for (obs_it = segmentation_list_.begin(); obs_it != segmentation_list_.end(); ++obs_it)
    {
      Segmentation& obs = *obs_it;
      // check if the segmentation is out of date... and if it is,
      // remove it and those that follow from the list
      if ((clock_->now() - obs.cloud_->header.stamp) > observation_keep_time_)
      {
        segmentation_list_.erase(obs_it, segmentation_list_.end());
        return;
      }
    }
  }
}

void SegmentationBuffer::updateClassMap(std::string new_class, uint8_t new_cost)
{
  class_names_cost_map_[new_class] = new_cost;
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
