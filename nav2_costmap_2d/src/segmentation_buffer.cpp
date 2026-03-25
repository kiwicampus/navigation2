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
#include <atomic>
#include <unordered_map>
#include <cmath>

#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "tf2/convert.h"
#include "rclcpp/rclcpp.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "visualization_msgs/msg/marker.hpp"
using namespace std::chrono_literals;

namespace nav2_costmap_2d {

namespace {

// --- DEBUG: single-tile latency WARN before pushObservation; delete this block when done ---
constexpr bool kDebugTileLatencyLog = true;
constexpr int kDebugTileLatencyIx = 0;
constexpr int kDebugTileLatencyIy = 0;
// Empty string = any buffer source; otherwise match e.g. "stereo"
constexpr const char kDebugTileLatencyBuffer[] = "";
// --- end DEBUG ---

/** Logs each logical step with delta since the previous tick; only RCLCPP_WARN (no snprintf). */
class StepDeltaTimer {
public:
  using Clock = std::chrono::high_resolution_clock;

  StepDeltaTimer(const rclcpp::Logger& logger, const char* buffer_source, uint64_t cloud_1based)
  : logger_(logger), buffer_(buffer_source), cloud_(cloud_1based), prev_(Clock::now())
  {
  }

  void step(const char* label)
  {
    const long delta_us = tick_delta_us();
    const double delta_ms = static_cast<double>(delta_us) / 1000.0;
    RCLCPP_WARN(
      logger_,
      "The step '%s' took %ld μs (%.3f ms) [cloud %llu, buffer '%s']",
      label, delta_us, delta_ms, static_cast<unsigned long long>(cloud_), buffer_);
  }

  void step_enter(const std::string& frame, unsigned width, unsigned height)
  {
    const long delta_us = tick_delta_us();
    const double delta_ms = static_cast<double>(delta_us) / 1000.0;
    RCLCPP_WARN(
      logger_,
      "The step 'enter' took %ld μs (%.3f ms) [cloud %llu, buffer '%s'] frame=%s width=%u height=%u",
      delta_us, delta_ms, static_cast<unsigned long long>(cloud_), buffer_, frame.c_str(), width,
      height);
  }

  void step_pixel_rows_batch(size_t v, size_t height)
  {
    const long delta_us = tick_delta_us();
    const double delta_ms = static_cast<double>(delta_us) / 1000.0;
    RCLCPP_WARN(
      logger_,
      "The step 'pixel_rows_batch' took %ld μs (%.3f ms) [cloud %llu, buffer '%s'] v=%zu of %zu",
      delta_us, delta_ms, static_cast<unsigned long long>(cloud_), buffer_, v, height);
  }

private:
  long tick_delta_us()
  {
    const auto now = Clock::now();
    const long delta_us =
      std::chrono::duration_cast<std::chrono::microseconds>(now - prev_).count();
    prev_ = now;
    return delta_us;
  }

  rclcpp::Logger logger_;
  const char* buffer_;
  uint64_t cloud_;
  Clock::time_point prev_;
};

}  // namespace

SegmentationBuffer::SegmentationBuffer(const nav2_util::LifecycleNode::WeakPtr& parent,
                                       std::string buffer_source, std::vector<std::string> class_types, std::unordered_map<std::string, CostHeuristicParams> class_names_cost_map, double observation_keep_time,
                                       double expected_update_rate, double max_lookahead_distance,
                                       double min_lookahead_distance, tf2_ros::Buffer& tf2_buffer,
                                       std::string global_frame, std::string sensor_frame,
                                       tf2::Duration tf_tolerance, double costmap_resolution, double tile_map_decay_time, bool visualize_tile_map, bool use_cost_selection,
                                       double camera_h_fov, double camera_v_fov, double camera_min_dist, double camera_max_dist,
                                       double fov_inside_decay_time, double fov_outside_decay_time, bool visualize_frustum_fov)
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
  , camera_h_fov_(camera_h_fov)
    , camera_v_fov_(camera_v_fov)
    , camera_min_dist_(camera_min_dist)
    , camera_max_dist_(camera_max_dist)
    , fov_inside_decay_time_(fov_inside_decay_time)
    , fov_outside_decay_time_(fov_outside_decay_time)
    , ground_fov_checker_(camera_h_fov, camera_v_fov, camera_min_dist, camera_max_dist)  // On init: 2D FOV checker
{
  RCLCPP_WARN(logger_, "SegmentationBuffer [%s]: Creating SegmentationBuffer", buffer_source_.c_str());
  auto node = parent.lock();
  clock_ = node->get_clock();
  logger_ = node->get_logger();
  last_updated_ = node->now();
  temporal_tile_map_ = std::make_shared<SegmentationTileMap>(costmap_resolution, tile_map_decay_time);
  visualize_tile_map_ = visualize_tile_map;
  use_cost_selection_ = use_cost_selection;
  RCLCPP_WARN(logger_, "SegmentationBuffer [%s]: Selection method = %s", 
              buffer_source_.c_str(), 
              use_cost_selection_ ? "COST-BASED (max_cost)" : "CONFIDENCE-BASED");
  visualize_frustum_fov_ = visualize_frustum_fov;
    if (visualize_frustum_fov_)
    {
        frustum_fov_pub_ = node->create_publisher<visualization_msgs::msg::Marker>(buffer_source + "/frustum_fov", 1);
    }
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
    const auto& name = semantic_class.class_name;
    if (class_names_cost_map_.find(name) == class_names_cost_map_.end()) {
      RCLCPP_WARN(
        logger_,
        "CRITICAL ERROR: Class '%s' from label_info is not defined in the costmap parameters! This class will be ignored.",
        name.c_str());
      continue;
    }
    class_to_id_map[name] = semantic_class.class_id;
  }
  segmentation_cost_multimap_ = std::make_shared<SegmentationCostMultimap>(class_to_id_map, class_names_cost_map_);
}

void SegmentationBuffer::bufferSegmentation(
  const sensor_msgs::msg::PointCloud2& cloud,
  const sensor_msgs::msg::Image& segmentation,
  const sensor_msgs::msg::Image& confidence)
{
  static std::atomic<uint64_t> cloud_counter{0};
  const uint64_t cloud_index = cloud_counter.fetch_add(1);
  const uint64_t cloud_1based = cloud_index + 1;

  const double buffer_entry_ros_s = clock_->now().seconds();
  const double cloud_header_stamp_s =
    static_cast<double>(cloud.header.stamp.sec) +
    static_cast<double>(cloud.header.stamp.nanosec) * 1e-9;

  RCLCPP_WARN(
    logger_,
    "[SegmentationBuffer DEBUG] [%s] heartbeat: processed %llu clouds",
    buffer_source_.c_str(),
    static_cast<unsigned long long>(cloud_1based));
 
  StepDeltaTimer timing(logger_, buffer_source_.c_str(), cloud_1based);
  timing.step_enter(
    cloud.header.frame_id, static_cast<unsigned>(segmentation.width),
    static_cast<unsigned>(segmentation.height));

  geometry_msgs::msg::PointStamped global_origin;
  std::string origin_frame = sensor_frame_ == "" ? cloud.header.frame_id : sensor_frame_;

  try {
    geometry_msgs::msg::PointStamped local_origin;
    local_origin.header.stamp = cloud.header.stamp;
    local_origin.header.frame_id = origin_frame;
    local_origin.point.x = 0;
    local_origin.point.y = 0;
    local_origin.point.z = 0;
    tf2_buffer_.transform(local_origin, global_origin, global_frame_, tf_tolerance_);
    timing.step("1_tf_transform_sensor_origin_to_global");

    geometry_msgs::msg::TransformStamped cam_tf =
      tf2_buffer_.lookupTransform(global_frame_, origin_frame, cloud.header.stamp, tf_tolerance_);
    timing.step("2_tf_lookup_sensor_in_global");

    if (fov_outside_decay_time_ > 0.0) {
    geometry_msgs::msg::Point frustum_origin;
    frustum_origin.x = global_origin.point.x;
    frustum_origin.y = global_origin.point.y;
    frustum_origin.z = global_origin.point.z;

    ground_fov_checker_.updatePose(frustum_origin, cam_tf.transform.rotation);
    timing.step("fov_checker_update_pose");
    }

    if (visualize_frustum_fov_ && frustum_fov_pub_) {
        std::vector<geometry_msgs::msg::Point> polygon =
      ground_fov_checker_.getGroundPolygonForVisualization();
      timing.step("fov_get_ground_polygon");
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = global_frame_;
      marker.header.stamp = cloud.header.stamp;
      marker.ns = buffer_source_ + "_frustum";
      marker.id = 0;
      marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.scale.x = 0.05;
      marker.color.r = 0.0f;
      marker.color.g = 1.0f;
      marker.color.b = 0.0f;
      marker.color.a = 1.0f;
      if (polygon.size() >= 3) {
        for (const auto& p : polygon) {
          marker.points.push_back(p);
        }
        marker.points.push_back(polygon.front());
      } else if (polygon.size() == 2) {
        marker.points.push_back(polygon[0]);
        marker.points.push_back(polygon[1]);
      }
      frustum_fov_pub_->publish(marker);
      timing.step("fov_frustum_marker_publish");
    }

    sensor_msgs::msg::PointCloud2 global_frame_cloud;
    tf2_buffer_.transform(cloud, global_frame_cloud, global_frame_, tf_tolerance_);
    timing.step("3_transform_pointcloud_to_global");

    global_frame_cloud.header.stamp = cloud.header.stamp;

    sensor_msgs::PointCloud2ConstIterator<float> iter_x_global(global_frame_cloud, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y_global(global_frame_cloud, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z_global(global_frame_cloud, "z");
    std::unordered_map<TileIndex, int> best_observations_idxs;
    double cloud_time_seconds =
      rclcpp::Time(cloud.header.stamp.sec, cloud.header.stamp.nanosec).seconds();
    RCLCPP_WARN(logger_, "cloud_time_seconds: %f", cloud_time_seconds);
    //timing.step("iterators_and_cloud_time_ready");

    for (size_t v = 0; v < segmentation.height; v++) {
      for (size_t u = 0; u < segmentation.width; u++) {
        int pixel_idx = v * segmentation.width + u;
        if (!std::isfinite(*(iter_z_global))) {
          ++iter_x_global;
          ++iter_y_global;
          ++iter_z_global;
          continue;
        }
        double sq_dist =
          std::pow(*(iter_x_global) - global_origin.point.x, 2) +
          std::pow(*(iter_y_global) - global_origin.point.y, 2) +
          std::pow(*(iter_z_global) - global_origin.point.z, 2);
        
        if (sq_dist >= sq_max_lookahead_distance_ || sq_dist <= sq_min_lookahead_distance_) {
          ++iter_x_global;
          ++iter_y_global;
          ++iter_z_global;
          continue;
        }

        TileIndex costmap_index = temporal_tile_map_->worldToIndex(*iter_x_global, *iter_y_global);

        auto it = best_observations_idxs.find(costmap_index);
        if (it != best_observations_idxs.end()) {
          if (use_cost_selection_) {
            uint8_t current_class = segmentation.data[pixel_idx];
            uint8_t existing_class = segmentation.data[it->second];
            auto current_cost = segmentation_cost_multimap_->getCostById(current_class);
            auto existing_cost = segmentation_cost_multimap_->getCostById(existing_class);
            if (current_cost.max_cost > existing_cost.max_cost) {
              best_observations_idxs[costmap_index] = pixel_idx;
              // RCLCPP_WARN(
              //   logger_,
              //   "COST-BASED: Replaced tile observation - current_class=%d (max_cost=%d) > "
              //   "existing_class=%d (max_cost=%d)",
              //   current_class, current_cost.max_cost, existing_class, existing_cost.max_cost);
            }
          } else {
            if (confidence.data[pixel_idx] > confidence.data[it->second]) {
              best_observations_idxs[costmap_index] = pixel_idx;
              // RCLCPP_WARN(
              //   logger_,
              //   "CONFIDENCE-BASED: Replaced tile observation - current_confidence=%d > "
              //   "existing_confidence=%d",
              //   confidence.data[pixel_idx], confidence.data[it->second]);
            }
          }
        } else {
          best_observations_idxs[costmap_index] = pixel_idx;
        }
        ++iter_x_global;
        ++iter_y_global;
        ++iter_z_global;
      }
    }
    timing.step("4_pixel_for_loop_complete");
    RCLCPP_WARN(logger_, "Processed %d pixels", segmentation.width * segmentation.height);

    temporal_tile_map_->lock();
    timing.step("5_tile_map_locked");

    if (fov_outside_decay_time_ > 0.0) {
      const double inside_decay =
        (fov_inside_decay_time_ > 0.0) ? fov_inside_decay_time_ : temporal_tile_map_->getDecayTime();
      const double outside_decay = fov_outside_decay_time_;
      int tiles_inside = 0, tiles_outside = 0;

      for (auto& tile : *temporal_tile_map_) {
        TileWorldXY world = temporal_tile_map_->indexToWorld(tile.first.x, tile.first.y);
        const bool inside = ground_fov_checker_.isInFOV(world.x, world.y);
        tile.second.setDecayTime(inside ? inside_decay : outside_decay);
        inside ? ++tiles_inside : ++tiles_outside;
      }
      RCLCPP_WARN(
        logger_,
        "SegmentationBuffer [%s] FOV decay applied: %d tiles inside (%.2fs), %d tiles outside (%.2fs)",
        buffer_source_.c_str(), tiles_inside, inside_decay, tiles_outside, outside_decay);
      timing.step("fov_decay_per_tile_done");
    } 
    temporal_tile_map_->purgeOldObservations(cloud_time_seconds);
    timing.step("6_purge_old_observations_in_buffer_segmentation");

    for (auto& idx : best_observations_idxs) {
      int img_idx_for_best_obs = idx.second;
      TileIndex costmap_index = idx.first;
      uint8_t class_id = segmentation.data[img_idx_for_best_obs];

      if (segmentation_cost_multimap_->hasClassId(class_id)) {
        TileObservation best_obs{
          class_id, static_cast<float>(confidence.data[img_idx_for_best_obs]), cloud_time_seconds};
        bool dominant_priority = segmentation_cost_multimap_->getCostById(class_id).dominant_priority;
        const bool buffer_ok =
          (kDebugTileLatencyBuffer[0] == '\0') || (buffer_source_ == kDebugTileLatencyBuffer);
        if (kDebugTileLatencyLog && buffer_ok && costmap_index.x == kDebugTileLatencyIx &&
            costmap_index.y == kDebugTileLatencyIy)
        {
          const double push_ros_s = clock_->now().seconds();
          RCLCPP_WARN(
            logger_,
            "[TileLatency] buffer=%s tile=(%d,%d) cloud_n=%llu "
            "buffer_entry_ros_s=%.6f cloud_header_stamp_s=%.6f push_ros_s=%.6f "
            "d_entry_to_push_s=%.6f d_stamp_to_push_s=%.6f obs_timestamp_s=%.6f class_id=%u",
            buffer_source_.c_str(), costmap_index.x, costmap_index.y,
            static_cast<unsigned long long>(cloud_1based), buffer_entry_ros_s, cloud_header_stamp_s,
            push_ros_s, push_ros_s - buffer_entry_ros_s, push_ros_s - cloud_header_stamp_s,
            cloud_time_seconds, static_cast<unsigned>(class_id));
        }
        temporal_tile_map_->pushObservation(best_obs, costmap_index, dominant_priority);
      } else {
        RCLCPP_WARN(
          logger_,
          "SegmentationBuffer [%s]: Skipping undefined class_id %d in tile (%d, %d)",
          buffer_source_.c_str(), class_id, costmap_index.x, costmap_index.y);
      }
    }
    timing.step("7_push_observations");

    temporal_tile_map_->unlock();
    timing.step("8_tile_map_unlocked");

    if (visualize_tile_map_) {
      sensor_msgs::msg::PointCloud2 tile_map_cloud =
        visualizeTemporalTileMap(*temporal_tile_map_, global_frame_, clock_->now());
      tile_map_pub_->publish(tile_map_cloud);
      timing.step("9_tile_map_visualization_publish");
    } else {
      timing.step("9_tile_map_visualization_skipped");
    }

    timing.step("10_bufferSegmentation_ok");
  } catch (const tf2::TransformException& ex) {
    timing.step("FAILED_tf_transform_exception");
    RCLCPP_ERROR(
      logger_,
      "TF Exception that should never happen for sensor frame: %s, cloud frame: %s, %s",
      sensor_frame_.c_str(), cloud.header.frame_id.c_str(), ex.what());
    return;
  } 
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