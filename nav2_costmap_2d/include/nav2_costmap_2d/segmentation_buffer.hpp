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
#ifndef NAV2_COSTMAP_2D__SEGMENTATION_BUFFER_HPP_
#define NAV2_COSTMAP_2D__SEGMENTATION_BUFFER_HPP_

#include <list>
#include <string>
#include <vector>

#include "nav2_costmap_2d/segmentation.hpp"
#include "nav2_util/lifecycle_node.hpp"
#include "rclcpp/time.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_sensor_msgs/tf2_sensor_msgs.hpp"
#include "vision_msgs/msg/label_info.hpp"

struct TileIndex {
    int x, y;

    bool operator==(const TileIndex& other) const {
        return x == other.x && y == other.y;
    }
};

namespace std {
    template<>
    struct hash<TileIndex> {
        size_t operator()(const TileIndex& coord) const {
            // Compute individual hash values for two integers
            // and combine them using bitwise XOR
            // and bit shifting:
            return std::hash<int>()(coord.x) ^ (std::hash<int>()(coord.y) << 1);
        }
    };
}

struct TileWorldXY
{
    double x, y;
};

struct TileObservation {
    uint8_t class_id, class_cost;
    float confidence;
    double timestamp;
};

class TemporalObservationQueue {
private:

    std::queue<std::unique_ptr<TileObservation>> queue_;
    float confidence_sum_ = 0.0f;
    double decay_time_;

public:
    TemporalObservationQueue(){}

    // Add an element with the current timestamp
    void push(std::unique_ptr<TileObservation> tile_obs_ptr) {
        if(tile_obs_ptr->class_id != getClassId())
        {
            std::queue<std::unique_ptr<TileObservation>> emptyQueue;
            std::swap(queue_, emptyQueue);
            confidence_sum_ = 0.0;
        }
        confidence_sum_ += tile_obs_ptr->confidence;
        queue_.emplace(std::move(tile_obs_ptr));
    }

    // Remove the front element
    void pop() {
        if (!queue_.empty()) {
            confidence_sum_ -= queue_.front()->confidence;
            queue_.pop();
        }
    }

    bool empty() {return queue_.empty();}

    void setDecayTime(float decay_time)
    {
        decay_time_ = decay_time;
    }

    // Get the current sum of all elements
    float getConfidenceSum() const {
        return confidence_sum_;
    }

    uint8_t getClassId()
    {
        if(!queue_.empty())
        {
            return queue_.back()->class_id;
        }
        return 255;
    }

    uint8_t getClassCost()
    {
        if(!queue_.empty())
        {
            return queue_.back()->class_cost;
        }
        return 0;
    }

    // Remove elements older than a specified duration (in seconds)
    void purgeOld() {
        double newest_obs_stamp = queue_.back()->timestamp;

        while (!queue_.empty()) {
            double age = newest_obs_stamp - queue_.front()->timestamp;
            if (age > decay_time_) {
                pop();
            } else {
                break;
            }
        }
    }
};

class SegmentationTileMap {
    private:
        std::unordered_map<TileIndex, TemporalObservationQueue> tile_map_;
        float resolution_;
        float decay_time_;

    public:
        // Define iterator types
        using Iterator = typename std::unordered_map<TileIndex, TemporalObservationQueue>::iterator;
        using ConstIterator = typename std::unordered_map<TileIndex, TemporalObservationQueue>::const_iterator;

        SegmentationTileMap(float resolution, float decay_time) : resolution_(resolution), decay_time_(decay_time) {}
        SegmentationTileMap(){}

        // Return iterator to the beginning of the tile_map_
        Iterator begin() { return tile_map_.begin(); }
        ConstIterator begin() const { return tile_map_.begin(); }

        // Return iterator to the end of the tile_map_
        Iterator end() { return tile_map_.end(); }
        ConstIterator end() const { return tile_map_.end(); }

        int size()
        {
            return tile_map_.size();
        }

        TileIndex worldToIndex(double x, double y) const {
            // Convert world coordinates to grid indices
            int ix = static_cast<int>(std::floor(x / resolution_));
            int iy = static_cast<int>(std::floor(y / resolution_));
            return TileIndex{ix, iy};
        }

        TileWorldXY indexToWorld(TileIndex idx) const {
            // Calculate the world coordinates of the center of the grid cell
            double x = (static_cast<double>(idx.x) + 0.5) * resolution_;
            double y = (static_cast<double>(idx.y) + 0.5) * resolution_;
            return TileWorldXY{x, y};
        }

        void pushObservation(TileObservation& obs, TileIndex& idx)
        {
            auto it = tile_map_.find(idx);
            if (it != tile_map_.end()) {
                // TileIndex exists, push the observation
                it->second.push(std::move(std::make_unique<TileObservation>(obs)));
            } else {
                // TileIndex does not exist, create a new TemporalObservationQueue with decay time
                TemporalObservationQueue& queue = tile_map_[idx];
                queue.setDecayTime(decay_time_);
                queue.push(std::move(std::make_unique<TileObservation>(obs)));
            }
        }

        void purgeOldObservations()
        {
            std::vector<TileIndex> tiles_to_remove;
            for (auto& tile : tile_map_)
            {
                tile.second.purgeOld();
                if(tile.second.empty())
                {
                    tiles_to_remove.emplace_back(tile.first);
                }
            }
            for (auto& tile : tiles_to_remove)
            {
                tile_map_.erase(tile);
            }
        }
};


namespace nav2_costmap_2d {
/**
 * @class SegmentationBuffer
 * @brief Takes in point clouds from sensors, transforms them to the desired frame, and stores them
 */
class SegmentationBuffer
{
   public:
    /**
     * @brief  Constructs an segmentation buffer
     * @param  topic_name The topic of the segmentations, used as an identifier for error and warning
     * messages
     * @param  observation_keep_time Defines the persistence of segmentations in seconds, 0 means only
     * keep the latest
     * @param  expected_update_rate How often this buffer is expected to be updated, 0 means there is
     * no limit
     * @param  min_obstacle_height The minimum height of a hitpoint to be considered legal
     * @param  max_obstacle_height The minimum height of a hitpoint to be considered legal
     * @param  obstacle_max_range The range to which the sensor should be trusted for inserting
     * obstacles
     * @param  obstacle_min_range The range from which the sensor should be trusted for inserting
     * obstacles
     * @param  raytrace_max_range The range to which the sensor should be trusted for raytracing to
     * clear out space
     * @param  raytrace_min_range The range from which the sensor should be trusted for raytracing to
     * clear out space
     * @param  tf2_buffer A reference to a tf2 Buffer
     * @param  global_frame The frame to transform PointClouds into
     * @param  sensor_frame The frame of the origin of the sensor, can be left blank to be read from
     * the messages
     * @param  tf_tolerance The amount of time to wait for a transform to be available when setting a
     * new global frame
     */
    SegmentationBuffer(const nav2_util::LifecycleNode::WeakPtr& parent, std::string buffer_source,
                       std::vector<std::string> class_types,
                       std::unordered_map<std::string, uint8_t> class_names_cost_map, double observation_keep_time,
                       double expected_update_rate, double max_lookahead_distance, double min_lookahead_distance,
                       tf2_ros::Buffer& tf2_buffer, std::string global_frame, std::string sensor_frame,
                       tf2::Duration tf_tolerance, double costmap_resolution);

    /**
     * @brief  Destructor... cleans up
     */
    ~SegmentationBuffer();

    /**
     * @brief  Transforms a PointCloud to the global frame and buffers it
     * <b>Note: The burden is on the user to make sure the transform is available... ie they should
     * use a MessageNotifier</b>
     * @param  cloud The cloud to be buffered
     */
    void bufferSegmentation(const sensor_msgs::msg::PointCloud2& cloud, const sensor_msgs::msg::Image& segmentation,
                            const sensor_msgs::msg::Image& confidence);

    /**
     * @brief  Pushes copies of all current segmentations onto the end of the vector passed in
     * @param  segmentations The vector to be filled
     */
    void getSegmentations(std::vector<Segmentation>& segmentations);

    /**
     * @brief  gets the class map associated with the segmentations stored in the buffer
     * @return the class map
     */
    std::unordered_map<std::string, uint8_t> getClassMap();

    void createClassIdCostMap(const vision_msgs::msg::LabelInfo& label_info);

    bool isClassIdCostMapEmpty() { return class_ids_cost_map_.empty(); }

    /**
     * @brief  Check if the segmentation buffer is being update at its expected rate
     * @return True if it is being updated at the expected rate, false otherwise
     */
    bool isCurrent() const;

    /**
     * @brief  Lock the segmentation buffer
     */
    inline void lock() { lock_.lock(); }

    /**
     * @brief  Lock the segmentation buffer
     */
    inline void unlock() { lock_.unlock(); }

    /**
     * @brief Reset last updated timestamp
     */
    void resetLastUpdated();

    /**
     * @brief Reset last updated timestamp
     */
    std::string getBufferSource() { return buffer_source_; }
    std::vector<std::string> getClassTypes() { return class_types_; }

    void setMinObstacleDistance(double distance) { sq_min_lookahead_distance_ = pow(distance, 2); }

    void setMaxObstacleDistance(double distance) { sq_max_lookahead_distance_ = pow(distance, 2); }

    void updateClassMap(std::string new_class, uint8_t new_cost);

    SegmentationTileMap* getSegmentationTileMap()
    {
        return &temporal_tile_map_;
    }

    uint8_t getCostForClassId(uint8_t class_id)
    {
        return class_ids_cost_map_[class_id];
    }

   private:
    /**
     * @brief  Removes any stale segmentations from the buffer list
     */
    void purgeStaleSegmentations();

    rclcpp::Clock::SharedPtr clock_;
    rclcpp::Logger logger_{rclcpp::get_logger("nav2_costmap_2d")};
    tf2_ros::Buffer& tf2_buffer_;
    std::vector<std::string> class_types_;
    std::unordered_map<std::string, uint8_t> class_names_cost_map_;
    std::unordered_map<uint16_t, uint8_t> class_ids_cost_map_;
    const rclcpp::Duration observation_keep_time_;
    const rclcpp::Duration expected_update_rate_;
    rclcpp::Time last_updated_;
    std::string global_frame_;
    std::string sensor_frame_;
    std::list<Segmentation> segmentation_list_;
    std::string buffer_source_;
    std::recursive_mutex lock_;  ///< @brief A lock for accessing data in callbacks safely
    double sq_max_lookahead_distance_;
    double sq_min_lookahead_distance_;
    tf2::Duration tf_tolerance_;

    SegmentationTileMap temporal_tile_map_;
};
}  // namespace nav2_costmap_2d
#endif  // NAV2_COSTMAP_2D__SEGMENTATION_BUFFER_HPP_
