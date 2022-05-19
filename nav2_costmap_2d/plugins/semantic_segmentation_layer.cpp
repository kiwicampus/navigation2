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
#include "nav2_costmap_2d/semantic_segmentation_layer.hpp"

#include "nav2_costmap_2d/costmap_math.hpp"
#include "nav2_costmap_2d/footprint.hpp"
#include "rclcpp/parameter_events_filter.hpp"

using nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE;
using nav2_costmap_2d::LETHAL_OBSTACLE;
using nav2_costmap_2d::NO_INFORMATION;

namespace nav2_costmap_2d {

SemanticSegmentationLayer::SemanticSegmentationLayer()
{
}

// This method is called at the end of plugin initialization.
// It contains ROS parameter(s) declaration and initialization
// of need_recalculation_ variable.
void SemanticSegmentationLayer::onInitialize()
{
    current_ = true;
    was_reset_ = false;
    auto node = node_.lock();
    if (!node)
    {
        throw std::runtime_error{"Failed to lock node"};
    }
    std::string segmentation_topic, camera_info_topic, pointcloud_topic, sensor_frame;
    std::vector<std::string> class_types_string;
    bool use_pointcloud;
    double max_lookahead_distance, min_lookahead_distance, observation_keep_time, transform_tolerance, expected_update_rate;

    declareParameter("enabled", rclcpp::ParameterValue(true));
    node->get_parameter(name_ + "." + "enabled", enabled_);
    declareParameter("combination_method", rclcpp::ParameterValue(1));
    node->get_parameter(name_ + "." + "combination_method", combination_method_);
    declareParameter("use_pointcloud", rclcpp::ParameterValue(false));
    node->get_parameter(name_ + "." + "use_pointcloud", use_pointcloud);
    declareParameter("max_lookahead_distance", rclcpp::ParameterValue(5.0));
    node->get_parameter(name_ + "." + "max_lookahead_distance", max_lookahead_distance);
    declareParameter("min_lookahead_distance", rclcpp::ParameterValue(0.3));
    node->get_parameter(name_ + "." + "min_lookahead_distance", min_lookahead_distance);
    declareParameter("segmentation_topic", rclcpp::ParameterValue(""));
    node->get_parameter(name_ + "." + "segmentation_topic", segmentation_topic);
    declareParameter("camera_info_topic", rclcpp::ParameterValue(""));
    node->get_parameter(name_ + "." + "camera_info_topic", camera_info_topic);
    declareParameter("pointcloud_topic", rclcpp::ParameterValue(""));
    node->get_parameter(name_ + "." + "pointcloud_topic", pointcloud_topic);
    declareParameter("observation_persistence", rclcpp::ParameterValue(0.0));
    node->get_parameter(name_ + "." + "observation_persistence", observation_keep_time);
    declareParameter(name_ + "." + "expected_update_rate", rclcpp::ParameterValue(0.0));
    node->get_parameter(name_  + "." + "sensor_frame", sensor_frame);
    node->get_parameter(
      name_ + "." + "expected_update_rate",
      expected_update_rate);
    node->get_parameter("transform_tolerance", transform_tolerance);

    declareParameter("class_types", rclcpp::ParameterValue(std::vector<std::string>({})));
    node->get_parameter(name_  + "." + "class_types", class_types_string);
    if(class_types_string.empty())
    {
        RCLCPP_ERROR(logger_, "no class types defined. Segmentation plugin cannot work this way");
        exit(-1);
    }

    for(auto& source : class_types_string)
    {
        std::vector<int64_t> classes_ids;
        uint8_t cost;
        declareParameter(source + ".classes", rclcpp::ParameterValue(std::vector<int64_t>({})));
        declareParameter(source + ".cost", rclcpp::ParameterValue(0));
        node->get_parameter(name_  + "." +source + ".classes", classes_ids);
        if(classes_ids.empty())
        {
            RCLCPP_ERROR(logger_, "no classes defined on type %s", source.c_str());
            continue;
        }
        node->get_parameter(name_  + "." +source + ".cost", cost);
        for(auto& class_id : classes_ids)
        {
            class_map_.insert(std::pair<uint8_t, uint8_t>((uint8_t)class_id, cost));
        }
    }

    if(class_map_.empty())
    {
        RCLCPP_ERROR(logger_, "No classes defined. Segmentation plugin cannot work this way");
        exit(-1);
    }


    global_frame_ = layered_costmap_->getGlobalFrameID();
    rolling_window_ = layered_costmap_->isRolling();

    matchSize();

    rmw_qos_profile_t custom_qos_profile = rmw_qos_profile_sensor_data;
    // semantic_segmentation_sub_ = node->create_subscription<vision_msgs::msg::SemanticSegmentation>(segmentation_topic, rclcpp::SensorDataQoS(), std::bind(&SemanticSegmentationLayer::segmentationCb, this, std::placeholders::_1));
    semantic_segmentation_sub_= std::make_shared<message_filters::Subscriber<vision_msgs::msg::SemanticSegmentation>>(rclcpp_node_, segmentation_topic, custom_qos_profile);
    pointcloud_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>(rclcpp_node_, pointcloud_topic, custom_qos_profile);
    pointcloud_tf_sub_ = std::make_shared<tf2_ros::MessageFilter<sensor_msgs::msg::PointCloud2>>(*pointcloud_sub_, *tf_, global_frame_, 50, rclcpp_node_, tf2::durationFromSec(transform_tolerance));
    segm_pc_sync_ = std::make_shared<message_filters::TimeSynchronizer<vision_msgs::msg::SemanticSegmentation, sensor_msgs::msg::PointCloud2>>(*semantic_segmentation_sub_, *pointcloud_tf_sub_, 100);
    segm_pc_sync_->registerCallback(std::bind(&SemanticSegmentationLayer::syncSegmPointcloudCb, this, std::placeholders::_1, std::placeholders::_2));
    
    observation_buffer_ = std::make_shared<nav2_costmap_2d::SegmentationBuffer>(node, pointcloud_topic, observation_keep_time, expected_update_rate,
        max_lookahead_distance,
        min_lookahead_distance,
        *tf_,
        global_frame_,
        sensor_frame, tf2::durationFromSec(transform_tolerance));

    sgm_debug_pub_ = node->create_publisher<vision_msgs::msg::SemanticSegmentation>("/buffered_segmentation", 1);
    orig_pointcloud_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("/buffered_pointcloud", 1);
    proc_pointcloud_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("/processed_pointcloud", 1);
    // ray_caster_.initialize(rclcpp_node_, camera_info_topic, pointcloud_topic, use_pointcloud, tf_, global_frame_, tf2::Duration(0), max_lookahead_distance);


    // msg_buffer_ = std::make_shared<ObjectBuffer<MessageTf>>(node, observation_keep_time);
    // msg_pc_buffer_ = std::make_shared<ObjectBuffer<MessagePointcloud>>(node, observation_keep_time);
}

// The method is called to ask the plugin: which area of costmap it needs to update.
// Inside this method window bounds are re-calculated if need_recalculation_ is true
// and updated independently on its value.
void SemanticSegmentationLayer::updateBounds(double robot_x, double robot_y, double /*robot_yaw*/,
                                             double* min_x, double* min_y, double* max_x, double* max_y)
{
    if (rolling_window_) {
        updateOrigin(robot_x - getSizeInMetersX() / 2, robot_y - getSizeInMetersY() / 2);
    }
    if (!enabled_) {
        current_ = true;
        return;
    } 
    // std::vector<MessageTf> segmentations;
    // std::vector<MessagePointcloud> segmentations2;
    // msg_buffer_->getObjects(segmentations);
    // msg_pc_buffer_->getObjects(segmentations2);
    std::vector<nav2_costmap_2d::Observation> observations;
    observation_buffer_->lock();
    observation_buffer_->getObservations(observations);
    observation_buffer_->unlock();
    int processed_msgs = 0;
    for(auto& observation : observations){
        proc_pointcloud_pub_->publish(*observation.cloud_);
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*observation.cloud_, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*observation.cloud_, "y");
        // sensor_msgs::PointCloud2ConstIterator<float> iter_z(*observation.cloud_, "z");
        sensor_msgs::PointCloud2ConstIterator<uint8_t> iter_class(*observation.cloud_, "class");
        // sensor_msgs::PointCloud2ConstIterator<uint8_t> iter_confidence(*observation.cloud_, "confidence");
        for(size_t point = 0; point < observation.cloud_->height*observation.cloud_->width; point++)
        {
            unsigned int mx, my;
            if(!worldToMap(*(iter_x+point), *(iter_y+point), mx, my))
            {
                RCLCPP_DEBUG(logger_, "Computing map coords failed");
                continue;
            }
            unsigned int index = getIndex(mx, my);
            uint8_t class_id = *(iter_class+point);
            if(!class_map_.count(class_id))
            {
                RCLCPP_DEBUG(logger_, "Cost for class id %i was not defined, skipping", class_id);
                continue;
            }
            costmap_[index] = class_map_[class_id];
            // if(*(iter_class+point)!= 1){
            //     costmap_[index] = nav2_costmap_2d::LETHAL_OBSTACLE;
            // }else{
            //     costmap_[index] = nav2_costmap_2d::FREE_SPACE;
            // }
            touch(*(iter_x+point), *(iter_y+point), min_x, min_y, max_x, max_y);
        }
        processed_msgs++;
    }
    if(processed_msgs>0)
    {
        RCLCPP_INFO(logger_, "Processed %i segmentations", processed_msgs);
    }
    
}

// The method is called when footprint was changed.
// Here it just resets need_recalculation_ variable.
void SemanticSegmentationLayer::onFootprintChanged()
{

    RCLCPP_DEBUG(rclcpp::get_logger("nav2_costmap_2d"),
                 "SemanticSegmentationLayer::onFootprintChanged(): num footprint points: %lu",
                 layered_costmap_->getFootprint().size());
}

// The method is called when costmap recalculation is required.
// It updates the costmap within its window bounds.
// Inside this method the costmap gradient is generated and is writing directly
// to the resulting costmap master_grid without any merging with previous layers.
void SemanticSegmentationLayer::updateCosts(nav2_costmap_2d::Costmap2D& master_grid, int min_i, int min_j, int max_i,
                                            int max_j)
{
    if (!enabled_)
    {
        return;
    }

    if (!current_ && was_reset_) {
        was_reset_ = false;
        current_ = true;
    }
    // (void) max_j;
    // (void) master_grid;
    // (void) min_i;
    // (void) max_i;
    // (void) min_j;
    if(!costmap_)
    {
        return;
    }
    // RCLCPP_INFO(logger_, "Updating costmap");
    switch (combination_method_) {
    case 0:  // Overwrite
      updateWithOverwrite(master_grid, min_i, min_j, max_i, max_j);
      break;
    case 1:  // Maximum
      updateWithMax(master_grid, min_i, min_j, max_i, max_j);
      break;
    default:  // Nothing
      break;
    }
}

void SemanticSegmentationLayer::segmentationCb(vision_msgs::msg::SemanticSegmentation::SharedPtr msg)
{
    std::cout << "not filter troll" << msg->header.stamp.sec << std::endl;
    geometry_msgs::msg::TransformStamped current_transform;
    try
    {
        current_transform = tf_->lookupTransform(global_frame_, msg->header.frame_id, msg->header.stamp);
    }
    catch (tf2::TransformException & ex) {
        // if an exception occurs, we need to remove the empty observation from the list
        RCLCPP_ERROR(
        logger_,
        "TF Exception that should never happen for sensor frame: %s, cloud frame: %s, %s",
        msg->header.frame_id.c_str(),
        global_frame_.c_str(), ex.what());
        return;
    }
    MessageTf message_tf;
    message_tf.message = *msg;
    message_tf.transform = current_transform;
    msg_buffer_->bufferObject(message_tf, msg->header.stamp);
    RCLCPP_INFO(logger_, "msg buffered");
}

void SemanticSegmentationLayer::segmentationCb2(const std::shared_ptr<const vision_msgs::msg::SemanticSegmentation> &msg){
    std::cout << "filter troll" << msg->header.stamp.sec << std::endl;
}

void SemanticSegmentationLayer::syncSegmPointcloudCb(const std::shared_ptr<const vision_msgs::msg::SemanticSegmentation> &segmentation, const std::shared_ptr<const sensor_msgs::msg::PointCloud2> &pointcloud)
{
    std::cout << "synctroll" << std::endl;
    if(segmentation->width*segmentation->height != pointcloud->width*pointcloud->height)
    {
        RCLCPP_WARN(logger_, "Pointcloud and segmentation sizes are different, will not buffer, WTF!");
        return;
    }
    unsigned expected_array_size = segmentation->width*segmentation->height;
    if(segmentation->data.size() < expected_array_size ||  segmentation->confidence.size() < expected_array_size)
    {
        RCLCPP_WARN(logger_, "segmentation arrays have wrong sizes: data->%lu, confidence->%lu, expected->%u", segmentation->data.size(), segmentation->confidence.size(), expected_array_size);
        return;
    }
    observation_buffer_->lock();
    observation_buffer_->bufferCloud(*pointcloud, *segmentation);
    observation_buffer_->unlock();
    sgm_debug_pub_->publish(*segmentation);
    orig_pointcloud_pub_->publish(*pointcloud);
}

void SemanticSegmentationLayer::reset()
{
    resetMaps();
    current_ = false;
    was_reset_ = true;
}

}  // namespace nav2_costmap_2d

// This is the macro allowing a nav2_costmap_2d::SemanticSegmentationLayer class
// to be registered in order to be dynamically loadable of base type nav2_costmap_2d::Layer.
// Usually places in the end of cpp-file where the loadable class written.
#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(nav2_costmap_2d::SemanticSegmentationLayer, nav2_costmap_2d::Layer)