/*********************************************************************
 *
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2008, 2013, Willow Garage, Inc.
 *  Copyright (c) 2020, Samsung R&D Institute Russia
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
 *         Alexey Merzlyakov
 *
 * Reference tutorial:
 * https://navigation.ros.org/tutorials/docs/writing_new_costmap2d_plugin.html
 *********************************************************************/
#ifndef SEMANTIC_SEGMENTATION_LAYER_HPP_
#define SEMANTIC_SEGMENTATION_LAYER_HPP_

#include "rclcpp/rclcpp.hpp"

#include "nav2_costmap_2d/layer.hpp"
#include "nav2_costmap_2d/costmap_layer.hpp"
#include "nav2_costmap_2d/layered_costmap.hpp"
#include "nav2_costmap_2d/ray_caster.hpp"
#include "nav2_costmap_2d/object_buffer.hpp"
#include "nav2_costmap_2d/segmentation_buffer.hpp"
#include "nav2_costmap_2d/observation.hpp"
#include "rclcpp/rclcpp.hpp"
#include "vision_msgs/msg/semantic_segmentation.hpp"
#include "opencv2/core.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/time_synchronizer.h"
#include "tf2_ros/message_filter.h"

namespace nav2_costmap_2d {

struct MessageTf 
{
    vision_msgs::msg::SemanticSegmentation message;
    geometry_msgs::msg::TransformStamped transform;
};

struct MessagePointcloud
{
    vision_msgs::msg::SemanticSegmentation message;
    sensor_msgs::msg::PointCloud2 original_pointcloud;
    sensor_msgs::msg::PointCloud2 world_frame_pointcloud;
};

class SemanticSegmentationLayer : public CostmapLayer
{
   public:
    SemanticSegmentationLayer();

    virtual void onInitialize();
    virtual void updateBounds(double robot_x, double robot_y, double robot_yaw, double* min_x, double* min_y,
                              double* max_x, double* max_y);
    virtual void updateCosts(nav2_costmap_2d::Costmap2D& master_grid, int min_i, int min_j, int max_i, int max_j);

    virtual void reset();

    virtual void onFootprintChanged();

    virtual bool isClearable() { return false; }

   private:
    void segmentationCb(vision_msgs::msg::SemanticSegmentation::SharedPtr msg);

    void segmentationCb2(const std::shared_ptr<const vision_msgs::msg::SemanticSegmentation> &msg);

    void syncSegmPointcloudCb(const std::shared_ptr<const vision_msgs::msg::SemanticSegmentation> &segmentation, const std::shared_ptr<const sensor_msgs::msg::PointCloud2> &pointcloud);

    RayCaster ray_caster_;

    std::shared_ptr<ObjectBuffer<MessageTf>> msg_buffer_;
    std::shared_ptr<ObjectBuffer<MessagePointcloud>> msg_pc_buffer_;

    // rclcpp::Subscription<vision_msgs::msg::SemanticSegmentation>::SharedPtr semantic_segmentation_sub_;
    std::shared_ptr<message_filters::Subscriber<vision_msgs::msg::SemanticSegmentation>> semantic_segmentation_sub_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> pointcloud_sub_;
    std::shared_ptr<message_filters::TimeSynchronizer<vision_msgs::msg::SemanticSegmentation, sensor_msgs::msg::PointCloud2>> segm_pc_sync_;
    std::shared_ptr<tf2_ros::MessageFilter<sensor_msgs::msg::PointCloud2>> pointcloud_tf_sub_;

    // debug publishers
    std::shared_ptr<rclcpp::Publisher<vision_msgs::msg::SemanticSegmentation>> sgm_debug_pub_;
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> orig_pointcloud_pub_;
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> proc_pointcloud_pub_;

    std::shared_ptr<nav2_costmap_2d::SegmentationBuffer> observation_buffer_;
    
    vision_msgs::msg::SemanticSegmentation latest_segmentation_message;

    std::string global_frame_;

    bool rolling_window_;
    bool was_reset_;
    int combination_method_;
};

}  // namespace nav2_costmap_2d

#endif  // SEMANTIC_SEGMENTATION_LAYER_HPP_