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
#include <cstdio>
#include <list>
#include <string>
#include <vector>

#include <array>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "tf2/convert.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "visualization_msgs/msg/marker.hpp"
using namespace std::chrono_literals;

namespace nav2_costmap_2d {
SegmentationBuffer::SegmentationBuffer(
    const nav2_util::LifecycleNode::WeakPtr& parent, std::string buffer_source, std::vector<std::string> class_types,
    std::unordered_map<std::string, CostHeuristicParams> class_names_cost_map, double observation_keep_time,
    double expected_update_rate, double max_lookahead_distance, double min_lookahead_distance,
    tf2_ros::Buffer& tf2_buffer, std::string global_frame, std::string sensor_frame, tf2::Duration tf_tolerance,
    double costmap_resolution, double tile_map_decay_time, bool visualize_tile_map, bool use_cost_selection,
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
    auto node = parent.lock();
    clock_ = node->get_clock();
    logger_ = node->get_logger();
    last_updated_ = node->now();
    temporal_tile_map_ = std::make_shared<SegmentationTileMap>(costmap_resolution, tile_map_decay_time);
    visualize_tile_map_ = visualize_tile_map;
    use_cost_selection_ = use_cost_selection;
    visualize_frustum_fov_ = visualize_frustum_fov;
    if (visualize_frustum_fov_)
    {
        frustum_fov_pub_ = node->create_publisher<visualization_msgs::msg::Marker>(buffer_source + "/frustum_fov", 1);
    }
    RCLCPP_INFO(logger_,
                "SegmentationBuffer [%s] started:\n"
                "  selection method:       %s\n"
                "  tile_map_decay_time:    %.2f s  (global, applied to every new tile)\n"
                "  fov_decay_time:         %.2f s  (-1 = use tile_map_decay_time)\n"
                "  outside_fov_decay_time: %.2f s  (-1 = FOV-aware decay disabled)",
                buffer_source_.c_str(), use_cost_selection_ ? "COST-BASED (max_cost)" : "CONFIDENCE-BASED",
                temporal_tile_map_->getDecayTime(), fov_inside_decay_time_, fov_outside_decay_time_);
    if (visualize_tile_map_)
    {
        tile_map_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>(buffer_source + "/tile_map", 1);
    }
}

SegmentationBuffer::~SegmentationBuffer() {}

void SegmentationBuffer::createSegmentationCostMultimap(const vision_msgs::msg::LabelInfo& label_info)
{
    std::unordered_map<std::string, uint8_t> class_to_id_map;
    for (const auto& semantic_class : label_info.class_map)
    {
        const auto& name = semantic_class.class_name;
        if (class_names_cost_map_.find(name) == class_names_cost_map_.end())
        {
            RCLCPP_INFO(logger_,
                        "CRITICAL ERROR: Class '%s' from label_info is not defined in the costmap parameters! This "
                        "class will be ignored.",
                        name.c_str());
            continue;
        }
        class_to_id_map[name] = semantic_class.class_id;
    }
    segmentation_cost_multimap_ = std::make_shared<SegmentationCostMultimap>(class_to_id_map, class_names_cost_map_);
}

void SegmentationBuffer::bufferSegmentation(const sensor_msgs::msg::PointCloud2& cloud,
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

        // Each costmap update cycle, after getting TF: update 2D FOV checker pose in global frame
        geometry_msgs::msg::TransformStamped cam_tf =
            tf2_buffer_.lookupTransform(global_frame_, origin_frame, cloud.header.stamp, tf_tolerance_);
        geometry_msgs::msg::Point frustum_origin;
        frustum_origin.x = global_origin.point.x;
        frustum_origin.y = global_origin.point.y;
        frustum_origin.z = global_origin.point.z;

        RCLCPP_INFO_THROTTLE(logger_, *clock_, 2000,
                             "[%s] 1) Extracted current TF: translation (%.3f, %.3f, %.3f) rotation (qx=%.3f qy=%.3f "
                             "qz=%.3f qw=%.3f) frame %s -> %s",
                             buffer_source_.c_str(), cam_tf.transform.translation.x, cam_tf.transform.translation.y,
                             cam_tf.transform.translation.z, cam_tf.transform.rotation.x, cam_tf.transform.rotation.y,
                             cam_tf.transform.rotation.z, cam_tf.transform.rotation.w, cam_tf.header.frame_id.c_str(),
                             cam_tf.child_frame_id.c_str());
        RCLCPP_INFO_THROTTLE(logger_, *clock_, 2000,
                             "[%s] 2) Will transform point (%.3f, %.3f, %.3f) to obtain the 4 points of the polygon",
                             buffer_source_.c_str(), frustum_origin.x, frustum_origin.y, frustum_origin.z);

        ground_fov_checker_.updatePose(frustum_origin, cam_tf.transform.rotation);
        last_frustum_origin_x_ = frustum_origin.x;
        last_frustum_origin_y_ = frustum_origin.y;
        last_frustum_origin_z_ = frustum_origin.z;

        std::vector<geometry_msgs::msg::Point> polygon = ground_fov_checker_.getGroundPolygonForVisualization();
        if (!polygon.empty())
        {
            std::string poly_str;
            for (size_t i = 0; i < polygon.size(); ++i)
            {
                char buf[64];
                std::snprintf(buf, sizeof(buf), " [%zu](%.3f,%.3f)", i, polygon[i].x, polygon[i].y);
                poly_str += buf;
            }
            RCLCPP_INFO_THROTTLE(logger_, *clock_, 2000, "[%s] 3b) Polygon vertices (x,y):%s",
                                buffer_source_.c_str(), poly_str.c_str());
        }

        if (visualize_frustum_fov_ && frustum_fov_pub_)
        {
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
            if (polygon.size() >= 3)
            {
                for (const auto& p : polygon)
                    marker.points.push_back(p);
                marker.points.push_back(polygon.front());  // close the polygon
            }
            else if (polygon.size() == 2)
            {
                // Only 2 rays hit z=0 (other 2 "above horizon"); draw the segment so something is visible
                marker.points.push_back(polygon[0]);
                marker.points.push_back(polygon[1]);
            }
            // else: 0 or 1 point -> leave points empty (frustum not visible on ground)
            frustum_fov_pub_->publish(marker);
        }

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
                double sq_dist = std::pow(*(iter_x_global)-global_origin.point.x, 2) +
                                 std::pow(*(iter_y_global)-global_origin.point.y, 2) +
                                 std::pow(*(iter_z_global)-global_origin.point.z, 2);
                if (sq_dist >= sq_max_lookahead_distance_ || sq_dist <= sq_min_lookahead_distance_)
                {
                    ++iter_x_global;
                    ++iter_y_global;
                    ++iter_z_global;
                    continue;
                }

                TileIndex costmap_index = temporal_tile_map_->worldToIndex(*iter_x_global, *iter_y_global);

                // Selection policy per tile: cost-based (max_cost) or confidence-based
                auto it = best_observations_idxs.find(costmap_index);
                if (it != best_observations_idxs.end())
                {
                    if (use_cost_selection_)
                    {
                        // Cost-based: pick highest max_cost
                        uint8_t current_class = segmentation.data[pixel_idx];
                        uint8_t existing_class = segmentation.data[it->second];
                        auto current_cost = segmentation_cost_multimap_->getCostById(current_class);
                        auto existing_cost = segmentation_cost_multimap_->getCostById(existing_class);
                        if (current_cost.max_cost > existing_cost.max_cost)
                        {
                            best_observations_idxs[costmap_index] = pixel_idx;
                            RCLCPP_DEBUG(logger_,
                                         "COST-BASED: Replaced tile observation - current_class=%d (max_cost=%d) > "
                                         "existing_class=%d (max_cost=%d)",
                                         current_class, current_cost.max_cost, existing_class, existing_cost.max_cost);
                        }
                    }
                    else
                    {
                        // Confidence-based: pick highest confidence
                        if (confidence.data[pixel_idx] > confidence.data[it->second])
                        {
                            best_observations_idxs[costmap_index] = pixel_idx;
                            RCLCPP_DEBUG(logger_,
                                         "CONFIDENCE-BASED: Replaced tile observation - current_confidence=%d > "
                                         "existing_confidence=%d",
                                         confidence.data[pixel_idx], confidence.data[it->second]);
                        }
                    }
                }
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

        // FOV-aware decay: tiles outside the current camera frustum decay faster.
        // Only applied when fov_outside_decay_time_ > 0 (feature explicitly enabled).
        if (fov_outside_decay_time_ > 0.0)
        {
            const double inside_decay =
                (fov_inside_decay_time_ > 0.0) ? fov_inside_decay_time_ : temporal_tile_map_->getDecayTime();
            const double outside_decay = fov_outside_decay_time_;
            int tiles_inside = 0, tiles_outside = 0;
            double sample_outside_world_x = 0.0, sample_outside_world_y = 0.0;
            double sample_inside_world_x = 0.0, sample_inside_world_y = 0.0;
            bool has_sample_outside = false, has_sample_inside = false;

            const size_t total_tiles = temporal_tile_map_->size();
            RCLCPP_INFO(logger_,
                        "SegmentationBuffer [%s] FOV decay enabled: fov_decay_time=%.2fs (inside), "
                        "outside_fov_decay_time=%.2fs, temporal_tile_map size=%zu",
                        buffer_source_.c_str(), inside_decay, outside_decay, total_tiles);

            if (polygon.empty())
            {
                RCLCPP_WARN_THROTTLE(logger_, *clock_, 5000,
                                    "SegmentationBuffer [%s] Frustum ground polygon is empty: no tiles will be "
                                    " Camera may be looking horizontal (rays do not hit z=0 in [min,max] range).",
                                    buffer_source_.c_str());
            }

            // For each (mx, my) tile in the map, test if its world position is inside the 2D FOV
            for (auto& tile : *temporal_tile_map_)
            {
                TileWorldXY world = temporal_tile_map_->indexToWorld(tile.first.x, tile.first.y);
                const bool inside = ground_fov_checker_.isInFOV(world.x, world.y);
                RCLCPP_INFO_THROTTLE(logger_, *clock_, 1000,
                                    "ANALYZING POINT [%s]: -> tile %d,%d world(%.2f, %.2f) |",
                                    buffer_source_.c_str(), tile.first.x, tile.first.y, world.x, world.y);
                if (!inside)
                {
                    if (!has_sample_outside)
                    {
                        sample_outside_world_x = world.x;
                        sample_outside_world_y = world.y;
                        has_sample_outside = true;
                    }
                    ++observations_outside_fov_count_;
                    // if (observations_outside_fov_count_ % 500 == 0)
                    // {
                    //     RCLCPP_INFO(logger_,
                    //                 "SegmentationBuffer [%s]: observation OUTSIDE FOV (every 500th: #%d) -> "
                    //                 "tile decay set to %.2fs | tile %d,%d world (%.2f, %.2f) |",
                    //                 buffer_source_.c_str(), observations_outside_fov_count_, outside_decay,
                    //                 tile.first.x, tile.first.y, world.x, world.y);
                    // }
                }
                else
                {
                    if (!has_sample_inside)
                    {
                        sample_inside_world_x = world.x;
                        sample_inside_world_y = world.y;
                        has_sample_inside = true;
                    }
                    ++observations_inside_fov_count_;
                    // if (observations_inside_fov_count_ % 500 == 0)
                    // {
                    //     RCLCPP_INFO(logger_,
                    //                 "SegmentationBuffer [%s]: observation INSIDE FOV (every 500th: #%d) -> "
                    //                 "tile decay set to %.2fs | tile %d,%d world (%.2f, %.2f) |",
                    //                 buffer_source_.c_str(), observations_inside_fov_count_, inside_decay, tile.first.x,
                    //                 tile.first.y, world.x, world.y);
                    // }
                }
                tile.second.setDecayTime(inside ? inside_decay : outside_decay);
                inside ? ++tiles_inside : ++tiles_outside;
            }
            RCLCPP_DEBUG(logger_,
                         "SegmentationBuffer [%s] FOV decay applied: %d tiles inside (%.2fs), %d tiles outside (%.2fs)",
                         buffer_source_.c_str(), tiles_inside, inside_decay, tiles_outside, outside_decay);
            RCLCPP_INFO_THROTTLE(
                logger_, *clock_, 5000,
                "SegmentationBuffer [%s] FOV summary: tiles_inside=%d tiles_outside=%d (running: inside=%d outside=%d)",
                buffer_source_.c_str(), tiles_inside, tiles_outside, observations_inside_fov_count_,
                observations_outside_fov_count_);
            if (tiles_inside == 0 && tiles_outside > 0 && has_sample_outside)
            {
                RCLCPP_WARN_THROTTLE(logger_, *clock_, 5000,
                                     "SegmentationBuffer [%s] No tiles inside FOV this update. "
                                     "Sample outside tile world (%.2f, %.2f).",
                                     buffer_source_.c_str(), sample_outside_world_x, sample_outside_world_y);
            }
            if (tiles_inside > 0 && has_sample_inside)
            {
                RCLCPP_INFO_THROTTLE(logger_, *clock_, 5000,
                                     "SegmentationBuffer [%s] Sample inside tile world (%.2f, %.2f)",
                                     buffer_source_.c_str(), sample_inside_world_x, sample_inside_world_y);
            }
        }

        temporal_tile_map_->purgeOldObservations(cloud_time_seconds);
        for (auto& idx : best_observations_idxs)
        {
            int img_idx_for_best_obs = idx.second;
            TileIndex costmap_index = idx.first;
            uint8_t class_id = segmentation.data[img_idx_for_best_obs];

            // Only process observations with defined class IDs
            if (segmentation_cost_multimap_->hasClassId(class_id))
            {   
                
                TileObservation best_obs{class_id, static_cast<float>(confidence.data[img_idx_for_best_obs]),
                                         cloud_time_seconds};
                RCLCPP_INFO_THROTTLE(logger_, *clock_, 1000, "SegmentationBuffer [%s]: Found observation for class_id %d in tile (%d, %d) with confidence %f",
                                            buffer_source_.c_str(), static_cast<int>(class_id), costmap_index.x, costmap_index.y,
                                            static_cast<double>(confidence.data[img_idx_for_best_obs]));
                bool dominant_priority = segmentation_cost_multimap_->getCostById(class_id).dominant_priority;
                temporal_tile_map_->pushObservation(best_obs, costmap_index, dominant_priority);
            }
            else
            {
                    RCLCPP_INFO_THROTTLE(logger_, *clock_, 1000, "SegmentationBuffer [%s]: Skipping undefined class_id %d in tile (%d, %d)",
                                buffer_source_.c_str(), class_id, costmap_index.x, costmap_index.y);
            }
        }
        temporal_tile_map_->unlock();

        if (visualize_tile_map_)
        {
            sensor_msgs::msg::PointCloud2 tile_map_cloud =
                visualizeTemporalTileMap(*temporal_tile_map_, global_frame_, clock_->now());
            tile_map_pub_->publish(tile_map_cloud);
        }

    } catch (tf2::TransformException& ex)
    {
        RCLCPP_ERROR(logger_, "TF Exception that should never happen for sensor frame: %s, cloud frame: %s, %s",
                     sensor_frame_.c_str(), cloud.header.frame_id.c_str(), ex.what());
        return;
    }

    // if the update was successful, we want to update the last updated time
    last_updated_ = clock_->now();
}

std::unordered_map<std::string, CostHeuristicParams> SegmentationBuffer::getClassMap() { return class_names_cost_map_; }

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
                    buffer_source_.c_str(), (clock_->now() - last_updated_).seconds(), expected_update_rate_.seconds());
    }
    return current;
}

void SegmentationBuffer::resetLastUpdated() { last_updated_ = clock_->now(); }
}  // namespace nav2_costmap_2d
