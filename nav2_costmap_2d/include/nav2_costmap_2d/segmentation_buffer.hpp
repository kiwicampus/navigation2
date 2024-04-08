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

#include "nav2_util/lifecycle_node.hpp"
#include "rclcpp/time.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_sensor_msgs/tf2_sensor_msgs.hpp"
#include "vision_msgs/msg/label_info.hpp"

/**
 * @brief Represents the parameters associated with the cost calculation for a given class
 */
struct CostHeuristicParams
{
 uint8_t base_cost, max_cost, mark_confidence;
 int samples_to_max_cost;
};

/**
 * @brief Represents a 2D grid index with equality comparison. Supports negative indexes
 */
struct TileIndex {
    int x, y;

    bool operator==(const TileIndex& other) const {
        return x == other.x && y == other.y;
    }
};

namespace std {
    /**
     * @brief Custom hash function for TileIndex to enable its use as a key in unordered_map.
     */
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


/**
 * @brief Represents the world coordinates of a tile.
 */
struct TileWorldXY
{
    double x, y;
};

/**
 * @brief Encapsulates the observation data for a tile, including class ID, cost, confidence, and timestamp.
 */
struct TileObservation {
    using UniquePtr = std::unique_ptr<TileObservation>;

    uint8_t class_id;
    float confidence;
    double timestamp;
};

/**
 * @brief Manages temporal observations with a decay mechanism, maintaining a sum of confidences.
 * Wraps a std::deque to store observations, allowing for efficient insertion and removal.
 */
class TemporalObservationQueue {
private:

    std::deque<TileObservation> queue_;
    float confidence_sum_ = 0.0f;
    double decay_time_;

public:
    TemporalObservationQueue(){}

    /**
     * @brief Adds an observation to the queue, resets the queue if class ID changes.
     * @param tile_obs The observation to add.
     */
    void push(TileObservation tile_obs) {
        if(tile_obs.class_id != getClassId())
        {
            std::deque<TileObservation> emptyQueue;
            std::swap(queue_, emptyQueue);
            confidence_sum_ = 0.0;
        }
        confidence_sum_ += tile_obs.confidence;
        queue_.push_back(tile_obs);
    }

    /**
     * @brief Removes the oldest observation from the queue.
     */
    void pop() {
        if (!queue_.empty()) {
            confidence_sum_ -= queue_.front().confidence;
            queue_.pop_front();
        }
    }

    /**
     * @brief Checks if the queue is empty.
     * @return True if empty, false otherwise.
     */
    bool empty() const {return queue_.empty();}

    /**
     * @brief Gets the size of the queue.
     * @return The number of observations in the queue.
     */
    int size() const { return queue_.size(); }

    /**
     * @brief Sets the decay time for observations.
     * @param decay_time The decay time in seconds.
     */
    void setDecayTime(float decay_time)
    {
        decay_time_ = decay_time;
    }

    /**
     * @brief Gets the current sum of confidence values of all observations.
     * @return The sum of confidences.
     */
    float getConfidenceSum() const {
        return confidence_sum_;
    }

    /**
     * @brief Gets the class ID of the most recent observation.
     * @return The class ID, or 0 if queue is empty.
     */
    uint8_t getClassId() const
    {
        if(!queue_.empty())
        {
            return queue_.back().class_id;
        }
        return 0;
    }

    /**
     * @brief Returns a copy of the deque object. Will have overhead
     * due to the copy operation but avoids race conditions since
     * the object in the class is not made editable by others
     * @return The class cost, or 0 if queue is empty.
     */
    std::deque<TileObservation> getQueue()
    {
        return queue_;
    } 

    /**
     * @brief Removes observations older than the decay time.
     * @param current_time The current time for comparison.
     */
    void purgeOld(double current_time) {
        while (!queue_.empty()) {
            double age = current_time - queue_.front().timestamp;
            if (age > decay_time_) {
                pop();
            } else {
                break;
            }
        }
    }
};

/**
 * @brief Manages a map of tile observations, allowing for spatial and temporal querying.
 * Utilizes an unordered_map to efficiently index observations by tile and supports locking for thread safety.
 */
class SegmentationTileMap {
    private:
        std::unordered_map<TileIndex, TemporalObservationQueue> tile_map_;
        float resolution_;
        float decay_time_;
        std::recursive_mutex lock_;


    public:
        using SharedPtr = std::shared_ptr<SegmentationTileMap>;

        // Define iterator types
        using Iterator = typename std::unordered_map<TileIndex, TemporalObservationQueue>::iterator;
        using ConstIterator = typename std::unordered_map<TileIndex, TemporalObservationQueue>::const_iterator;

        SegmentationTileMap(float resolution, float decay_time) : resolution_(resolution), decay_time_(decay_time) {
            tile_map_.reserve(1e4);
        }
        SegmentationTileMap(){}

        // Return iterator to the beginning of the tile_map_
        Iterator begin() { return tile_map_.begin(); }
        ConstIterator begin() const { return tile_map_.begin(); }

        // Return iterator to the end of the tile_map_
        Iterator end() { return tile_map_.end(); }
        ConstIterator end() const { return tile_map_.end(); }

        /**
         * @brief Locks the map for exclusive access.
         */
        inline void lock() { lock_.lock(); }

        /**
         * @brief Unlocks the map.
         */
        inline void unlock() { lock_.unlock(); }

        /**
         * @brief Returns the number of elements in the map.
         * @return The size of the map.
         */
        int size()
        {
            return tile_map_.size();
        }

        /**
         * @brief Converts world coordinates to a TileIndex.
         * @param x X coordinate in world space.
         * @param y Y coordinate in world space.
         * @return The corresponding TileIndex.
         */
        TileIndex worldToIndex(double x, double y) const {
            // Convert world coordinates to grid indices
            int ix = static_cast<int>(std::floor(x / resolution_));
            int iy = static_cast<int>(std::floor(y / resolution_));
            return TileIndex{ix, iy};
        }

        /**
         * @brief Converts a TileIndex to world coordinates.
         * @param idx The index to convert.
         * @return The world coordinates of the tile's center.
         */
        TileWorldXY indexToWorld(TileIndex idx) const {
            // Calculate the world coordinates of the center of the grid cell
            double x = (static_cast<double>(idx.x) + 0.5) * resolution_;
            double y = (static_cast<double>(idx.y) + 0.5) * resolution_;
            return TileWorldXY{x, y};
        }

        /**
         * @brief Adds an observation to the specified tile.
         * @param obs The observation to add.
         * @param idx The index of the tile.
         */
        void pushObservation(TileObservation& obs, TileIndex& idx)
        {
            auto it = tile_map_.find(idx);
            if (it != tile_map_.end()) {
                // TileIndex exists, push the observation
                it->second.push(obs);
            } else {
                // TileIndex does not exist, create a new TemporalObservationQueue with decay time
                TemporalObservationQueue& queue = tile_map_[idx];
                queue.setDecayTime(decay_time_);
                queue.push(obs);
            }
        }

        /**
         * @brief Removes observations older than the decay time from all tiles.
         * @param current_time The current time for comparison.
         */
        void purgeOldObservations(double current_time)
        {
            std::vector<TileIndex> tiles_to_remove;
            for (auto& tile : tile_map_)
            {
                tile.second.purgeOld(current_time);
                if(tile.second.empty())
                {
                    tiles_to_remove.emplace_back(tile.first);
                }
            }
            if(tile_map_.size() > 0)
            for (auto& tile : tiles_to_remove)
            {
                tile_map_.erase(tile);
            }
        }
};

struct PointData {
    float x, y, z;
    float confidence, confidence_sum;
    uint8_t class_id;
};

/**
 * @brief Creates a PointCloud2 message that contains a visual representation of 
 * a temporal tile map. There's a "column" of points on each tile, each point represents
 * a segmentation observation over that tile and they are all stacked together. Each observation
 * Has a channel for the class, for the confidence, and the confidence sum of the observations
 * over that tile
 * @param tileMap The segmentation tile map
 */
sensor_msgs::msg::PointCloud2 visualizeTemporalTileMap(SegmentationTileMap& tileMap) {
    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header.frame_id = "map";  // Set appropriate frame_id
    cloud.header.stamp = rclcpp::Clock().now();  // Set current time as timestamp

    // Define fields for PointCloud2
    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2Fields(6, "x", 1, sensor_msgs::msg::PointField::FLOAT32,
                                     "y", 1, sensor_msgs::msg::PointField::FLOAT32,
                                     "z", 1, sensor_msgs::msg::PointField::FLOAT32,
                                     "confidence", 1, sensor_msgs::msg::PointField::FLOAT32,
                                     "confidence_sum", 1, sensor_msgs::msg::PointField::FLOAT32,
                                     "class", 1, sensor_msgs::msg::PointField::UINT8);

    // Reserve space for points
    std::vector<PointData> points;
    for (auto& tile : tileMap) {
        TileIndex idx = tile.first;
        TileWorldXY worldXY = tileMap.indexToWorld(idx);
        double z = 0.0;
        for (auto& obs : tile.second.getQueue()) {
            PointData point;
            point.x = worldXY.x;
            point.y = worldXY.y;
            point.z = z;
            point.confidence = obs.confidence;
            point.confidence_sum = tile.second.getConfidenceSum() / tile.second.size();
            point.class_id = static_cast<uint8_t>(obs.class_id);
            points.push_back(point);
            z += 0.02;  // Increment Z by 0.02m for each observation
        }
    }

    // Set data in PointCloud2
    modifier.resize(points.size());  // Number of points
    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");
    sensor_msgs::PointCloud2Iterator<float> iter_confidence(cloud, "confidence");
    sensor_msgs::PointCloud2Iterator<float> iter_confidence_sum(cloud, "confidence_sum");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_class(cloud, "class");

    for (const auto& point : points) {
        *iter_x = point.x;
        *iter_y = point.y;
        *iter_z = point.z;
        *iter_confidence = point.confidence;
        *iter_confidence_sum = point.confidence_sum;
        *iter_class = point.class_id;
        ++iter_x; ++iter_y; ++iter_z; ++iter_confidence;++iter_confidence_sum; ++iter_class;
    }

    return cloud;
}

/**
 * Manages segmentation class information, including mapping between class names and IDs,
 * as well as managing the cost heuristic parameters associated with each class.
 */
class SegmentationCostMultimap {
public:
    SegmentationCostMultimap(){}
    /**
     * Constructs the SegmentationCostMultimap.
     * 
     * @param nameToIdMap A map from class names to class IDs.
     * @param nameToCostMap A map from class names to CostHeuristicParams.
     */
    SegmentationCostMultimap(const std::unordered_map<std::string, uint8_t>& nameToIdMap,
                             const std::unordered_map<std::string, CostHeuristicParams>& nameToCostMap) {
        for (const auto& pair : nameToIdMap) {
            const auto& name = pair.first;
            uint8_t id = pair.second;
            CostHeuristicParams cost = nameToCostMap.at(name);
            this->name_to_id_[name] = id;
            this->id_to_cost_[id] = cost;
        }
    }

    /**
     * Updates the cost heuristic parameters associated with a class ID.
     * 
     * @param id The class ID.
     * @param cost The new CostHeuristicParams to associate with the class.
     */
    void updateCostById(uint8_t id, const CostHeuristicParams& cost) {
        id_to_cost_[id] = cost;
    }

    /**
     * Retrieves the cost heuristic parameters associated with a class ID.
     * 
     * @param id The class ID.
     * @return The CostHeuristicParams associated with the class.
     */
    CostHeuristicParams getCostById(uint8_t id) const {
        return id_to_cost_.at(id);
    }

    /**
     * Updates the cost heuristic parameters associated with a class name.
     * 
     * @param name The class name.
     * @param cost The new CostHeuristicParams to associate with the class.
     */
    void updateCostByName(const std::string& name, const CostHeuristicParams& cost) {
        uint8_t id = name_to_id_.at(name);
        id_to_cost_[id] = cost;
    }

    /**
     * Retrieves the cost heuristic parameters associated with a class name.
     * 
     * @param name The class name.
     * @return The CostHeuristicParams associated with the class.
     */
    CostHeuristicParams getCostByName(const std::string& name) const {
        uint8_t id = name_to_id_.at(name);
        return id_to_cost_.at(id);
    }

    bool empty()
    {
        return name_to_id_.empty() || id_to_cost_.empty();
    }

private:
    std::unordered_map<std::string, uint8_t> name_to_id_; // Maps class names to class IDs
    std::unordered_map<uint8_t, CostHeuristicParams> id_to_cost_; // Maps class IDs to CostHeuristicParams
};

namespace nav2_costmap_2d {
/**
 * @class SegmentationBuffer
 * @brief Takes in point clouds from sensors, transforms them to the desired frame, and stores them
 */
class SegmentationBuffer
{
   public:
    using SharedPtr = std::shared_ptr<SegmentationBuffer>;
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
                       std::unordered_map<std::string, CostHeuristicParams> class_names_cost_map, double observation_keep_time,
                       double expected_update_rate, double max_lookahead_distance, double min_lookahead_distance,
                       tf2_ros::Buffer& tf2_buffer, std::string global_frame, std::string sensor_frame,
                       tf2::Duration tf_tolerance, double costmap_resolution, double tile_map_decay_time, bool visualize_tile_map = false);

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
     * @brief  gets the class map associated with the segmentations stored in the buffer
     * @return the class map
     */
    std::unordered_map<std::string, CostHeuristicParams> getClassMap();

    void createSegmentationCostMultimap(const vision_msgs::msg::LabelInfo& label_info);

    bool isClassIdCostMapEmpty() { return segmentation_cost_multimap_.empty(); }

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

    void updateClassMap(std::string new_class, CostHeuristicParams new_cost);

    SegmentationTileMap::SharedPtr getSegmentationTileMap()
    {
        return temporal_tile_map_;
    }

    CostHeuristicParams getCostForClassId(uint8_t class_id)
    {
        return segmentation_cost_multimap_.getCostById(class_id);
    }

    CostHeuristicParams getCostForClassName(std::string class_name)
    {
        return segmentation_cost_multimap_.getCostByName(class_name);
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
    std::unordered_map<std::string, CostHeuristicParams> class_names_cost_map_;
    const rclcpp::Duration observation_keep_time_;
    const rclcpp::Duration expected_update_rate_;
    rclcpp::Time last_updated_;
    std::string global_frame_;
    std::string sensor_frame_;
    std::string buffer_source_;
    std::recursive_mutex lock_;  ///< @brief A lock for accessing data in callbacks safely
    double sq_max_lookahead_distance_;
    double sq_min_lookahead_distance_;
    tf2::Duration tf_tolerance_;
    
    SegmentationCostMultimap segmentation_cost_multimap_;

    SegmentationTileMap::SharedPtr temporal_tile_map_;

    bool visualize_tile_map_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr tile_map_pub_;
};
}  // namespace nav2_costmap_2d
#endif  // NAV2_COSTMAP_2D__SEGMENTATION_BUFFER_HPP_
