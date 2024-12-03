#include "nav2_costmap_2d/lane_keeping_costmap_layer.hpp"
#include "nav2_costmap_2d/costmap_math.hpp"
#include <algorithm>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>


namespace nav2_costmap_2d
{

LaneKeepingLayer::LaneKeepingLayer()
: last_min_x_(-std::numeric_limits<float>::max()),
last_min_y_(-std::numeric_limits<float>::max()),
last_max_x_(std::numeric_limits<float>::max()),
last_max_y_(std::numeric_limits<float>::max())
{
}

LaneKeepingLayer::~LaneKeepingLayer()
{
}

void LaneKeepingLayer::onInitialize()
{
  current_ = true;
  was_reset_ = false;
  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error{"Failed to lock node"};
  }
  declareParameter("enabled", rclcpp::ParameterValue(true));
  declareParameter("keep_right", rclcpp::ParameterValue(true));
  declareParameter("max_distance", rclcpp::ParameterValue(5.0));       // in meters
  declareParameter("max_cost", rclcpp::ParameterValue(150.0));
  declareParameter("min_cost", rclcpp::ParameterValue(0.0));
  declareParameter("map_topic", rclcpp::ParameterValue("/map"));
  declareParameter("global_path_topic", rclcpp::ParameterValue("/nav2/cartesian_geopath"));
  declareParameter("global_odom_topic", rclcpp::ParameterValue("/odometry/global"));
  declareParameter("map_resolution", rclcpp::ParameterValue(0.1));
  getParameters();
  margin_ = static_cast<int>(10.0 / map_resolution_); // 10 meters of margin


  // Set up subscriptions
  odom_sub_ = node->create_subscription<nav_msgs::msg::Odometry>(
    global_odom_topic_, rclcpp::SensorDataQoS(),
    std::bind(&LaneKeepingLayer::odomCallback, this, std::placeholders::_1));

  path_sub_ = node->create_subscription<nav_msgs::msg::Path>(
    global_path_topic_, rclcpp::SystemDefaultsQoS(),
    std::bind(&LaneKeepingLayer::pathCallback, this, std::placeholders::_1));

  map_sub_ = node->create_subscription<nav_msgs::msg::OccupancyGrid>(
    map_topic_, rclcpp::QoS(10).transient_local().reliable().keep_last(1),
    std::bind(&LaneKeepingLayer::mapCallback, this, std::placeholders::_1));

  global_frame_ = layered_costmap_->getGlobalFrameID();
  rolling_window_ = layered_costmap_->isRolling();
  default_value_ = NO_INFORMATION;
  enabled_ = true;
  matchSize();
  dyn_params_handler_ = node->add_on_set_parameters_callback(
    std::bind(
      &LaneKeepingLayer::dynamicParametersCallback,
      this,
      std::placeholders::_1));


  RCLCPP_INFO(logger_, "LaneKeepingLayer::onInitialize() - Initialized");

}

void
LaneKeepingLayer::activate()
{
}

void
LaneKeepingLayer::deactivate()
{
  auto node = node_.lock();
}

void LaneKeepingLayer::reset()
{
  // return;
  resetMaps();
  current_ = false;
  was_reset_ = true;
}

void LaneKeepingLayer::getParameters()
{
  auto node = node_.lock();
  node->get_parameter(name_ + "." + "enabled", enabled_);
  node->get_parameter(name_ + "." + "keep_right", keep_right_);
  node->get_parameter(name_ + "." + "max_distance", max_distance_);
  node->get_parameter(name_ + "." + "max_cost", max_cost_);
  node->get_parameter(name_ + "." + "min_cost", min_cost_);
  node->get_parameter(name_ + "." + "map_topic", map_topic_);
  node->get_parameter(name_ + "." + "global_path_topic", global_path_topic_);
  node->get_parameter(name_ + "." + "global_odom_topic", global_odom_topic_);
  node->get_parameter(name_ + "." + "map_resolution", map_resolution_);

  // Print parameters
  RCLCPP_INFO(logger_, "LaneKeepingLayer::getParameters() - keep_right: %s", keep_right_ ? "true" : "false");
  RCLCPP_INFO(logger_, "LaneKeepingLayer::getParameters() - max_distance: %f", max_distance_);
  RCLCPP_INFO(logger_, "LaneKeepingLayer::getParameters() - max_cost: %f", max_cost_);
  RCLCPP_INFO(logger_, "LaneKeepingLayer::getParameters() - min_cost: %f", min_cost_);
  RCLCPP_INFO(logger_, "LaneKeepingLayer::getParameters() - map_topic: %s", map_topic_.c_str());
  RCLCPP_INFO(logger_, "LaneKeepingLayer::getParameters() - global_path_topic: %s", global_path_topic_.c_str());
  RCLCPP_INFO(logger_, "LaneKeepingLayer::getParameters() - global_odom_topic: %s", global_odom_topic_.c_str());
  RCLCPP_INFO(logger_, "LaneKeepingLayer::getParameters() - map_resolution: %f", map_resolution_);
}

rcl_interfaces::msg::SetParametersResult LaneKeepingLayer::dynamicParametersCallback(std::vector<rclcpp::Parameter> parameters)
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  auto result = rcl_interfaces::msg::SetParametersResult();
  for (auto parameter : parameters) {
    const auto & type = parameter.get_type();
    const auto & name = parameter.get_name();
    if (type == rclcpp::ParameterType::PARAMETER_BOOL) {
      if (name == name_ + "." + "enabled") {
        RCLCPP_WARN(logger_, "LaneKeepingLayer::dynamicParametersCallback() - enabled: %s", parameter.as_bool() ? "true" : "false");
        enabled_ = parameter.as_bool();
      }
    }
    if (type == rclcpp::ParameterType::PARAMETER_DOUBLE) {
      if (name == name_ + "." + "max_distance") {
        max_distance_ = parameter.as_double();
        RCLCPP_INFO(logger_, "LaneKeepingLayer::dynamicParametersCallback() - max_distance: %f", max_distance_);
      }
      if (name == name_ + "." + "max_cost") {
        max_cost_ = parameter.as_double();
        RCLCPP_INFO(logger_, "LaneKeepingLayer::dynamicParametersCallback() - max_cost: %f", max_cost_);
      }
      if (name == name_ + "." + "min_cost") {
        min_cost_ = parameter.as_double();
        RCLCPP_INFO(logger_, "LaneKeepingLayer::dynamicParametersCallback() - min_cost: %f", min_cost_);
      }
      if (name == name_ + "." + "map_resolution") {
        map_resolution_ = parameter.as_double();
      }
    }
    
  }
  result.successful = true;
  return result;
}

void LaneKeepingLayer::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  std::lock_guard<std::mutex> guard(data_mutex_);
  robot_pose_.header = msg->header;
  robot_pose_.pose = msg->pose.pose;
  robot_x_ = robot_pose_.pose.position.x;
  robot_y_ = robot_pose_.pose.position.y;
}

void LaneKeepingLayer::pathCallback(const nav_msgs::msg::Path::SharedPtr msg)
{
  std::lock_guard<std::mutex> guard(data_mutex_);
  global_path_ = msg;
  path_index_ = 0;
  if(static_cast<int>(global_path_->poses.size()) < 2) {
    RCLCPP_WARN(logger_, "LaneKeepingLayer::pathCallback() path_index_ is out of bounds");
    prev_goal_x_ = 0.0;
    prev_goal_y_ = 0.0;
    goal_x_ = 0.0;
    goal_y_ = 0.0;
    return;
  }
  prev_goal_x_ = global_path_->poses[path_index_].pose.position.x;
  prev_goal_y_ = global_path_->poses[path_index_].pose.position.y;
  goal_x_ = global_path_->poses[path_index_+1].pose.position.x;
  goal_y_ = global_path_->poses[path_index_+1].pose.position.y;
}

void LaneKeepingLayer::mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
{
  std::lock_guard<std::mutex> guard(data_mutex_);
  static_map_ = msg;
}


unsigned char LaneKeepingLayer::computeCost(float distance)
{
    if (distance > max_distance_)
    {
        return static_cast<unsigned char>(max_cost_);
    }
    else
    {
        // return static_cast<unsigned char>(min_cost + (max_cost - min_cost) * (1.0-std::exp(-distance / max_distance)));
        return static_cast<unsigned char>(min_cost_ + (max_cost_ - min_cost_) * distance / max_distance_);
    }
}


void LaneKeepingLayer::setCost(unsigned int mx, unsigned int my, unsigned char cost)
{
    // Check bounds and set cost
    if (mx < getSizeInCellsX() && my < getSizeInCellsY())
    {
        unsigned int index = getIndex(mx, my);
        costmap_[index] = cost;
    }

}

void LaneKeepingLayer::computeDistancesToWall(
    const std::vector<Eigen::Vector2f>& search_area,
    const std::vector<Eigen::Vector2f>& wall_points,
    std::vector<float>& distances)
{
    // Create a wall object to store the wall points
    Wall wall;
    wall.pts = wall_points;

    // Build KDTree index
    typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, Wall>,
        Wall,
        2 /* dimension */
    > my_kd_tree_t;

    my_kd_tree_t index(2 /*dim*/, wall, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    index.buildIndex();

    // Prepare the result vector
    distances.resize(search_area.size());

    // Query the nearest neighbor for each point in search_area
    for (size_t i = 0; i < search_area.size(); ++i)
    {
        float query_pt[2] = { search_area[i].x(), search_area[i].y() };
        size_t ret_index;
        float out_dist_sqr;
        nanoflann::KNNResultSet<float> resultSet(1);
        resultSet.init(&ret_index, &out_dist_sqr);
        index.findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));
        distances[i] = std::sqrt(out_dist_sqr);
        // Compute the cost for the current cell based on the distance to the wall
        unsigned char cost = computeCost(distances[i]);
        // Get the map coordinates of the current cell
        unsigned int mx, my;
        if (!worldToMap(search_area[i].x(), search_area[i].y(), mx, my))
        {
          RCLCPP_WARN(logger_, "Computing map coords failed: %f, %f, limits: %f, %f", search_area[i].x(), search_area[i].y(), getSizeInMetersX(), getSizeInMetersY());
          continue;
        }
        // Set the cost for the current cell
        setCost(mx, my, cost);
    }
}

void LaneKeepingLayer::updateBounds(double robot_x, double robot_y, double /*robot_yaw*/,
                                             double* min_x, double* min_y, double* max_x,
                                             double* max_y)
{
  // RCLCPP_WARN(logger_, "LaneKeepingLayer::updateBounds()");
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());

  if (!enabled_)
  {
    return;
  }
  if (rolling_window_)
  {
    updateOrigin(robot_x - getSizeInMetersX() / 2, robot_y - getSizeInMetersY() / 2);
  }

  double tmp_min_x = last_min_x_;
  double tmp_min_y = last_min_y_;
  double tmp_max_x = last_max_x_;
  double tmp_max_y = last_max_y_;
  last_min_x_ = *min_x;
  last_min_y_ = *min_y;
  last_max_x_ = *max_x;
  last_max_y_ = *max_y;
  *min_x = std::min(tmp_min_x, *min_x);
  *min_y = std::min(tmp_min_y, *min_y);
  *max_x = std::max(tmp_max_x, *max_x);
  *max_y = std::max(tmp_max_y, *max_y);

  current_ = true;
}

void LaneKeepingLayer::updateCosts(nav2_costmap_2d::Costmap2D& master_grid, int min_i,
                                            int min_j, int max_i, int max_j)
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  if (!enabled_)
  {
    return;
  }
  if (!current_ && was_reset_)
  {
    RCLCPP_WARN(logger_, "LaneKeepingLayer::updateCosts() called but not current and was reset");
    was_reset_ = false;
    current_ = true;
  }
  if(global_path_ == nullptr || global_path_->poses.size() < 2)
  {
    RCLCPP_WARN(logger_, "LaneKeepingLayer::updateCosts() global_path_ is null or has less than 2 poses");
    return;
  }

  // Check if the robot has reached the goal point and update the path index
  double distance_to_goal = distance(robot_x_, robot_y_, goal_x_, goal_y_);
  if(distance_to_goal < 6.0) {
    path_index_++;
    update_costmap_ = true;
  }
  if(path_index_ + 1 >= static_cast<int>(global_path_->poses.size())) {
      path_index_--;
  }
  goal_x_ = global_path_->poses[path_index_+1].pose.position.x;
  goal_y_ = global_path_->poses[path_index_+1].pose.position.y;

  if(update_costmap_) 
  {
    std::vector<Eigen::Vector2f> right_wall_points;
    std::vector<Eigen::Vector2f> search_area;
    // Iterate through waypoint pairs in the global path
    for (size_t idx = path_index_; idx < global_path_->poses.size() - 1; ++idx)
    {
      // Get start and goal waypoints
      double wx1 = global_path_->poses[idx].pose.position.x;
      double wy1 = global_path_->poses[idx].pose.position.y;
      double wx2 = global_path_->poses[idx+1].pose.position.x;
      double wy2 = global_path_->poses[idx+1].pose.position.y;

      // Check if at least the previous waypoint is within the costmap bounds
      unsigned int mx1, my1;
      if (!worldToMap(wx1, wy1, mx1, my1))
      {
        continue;
        RCLCPP_WARN(logger_, "LaneKeepingLayer::updateCosts() - first point out of bounds");
      }
      int mx2, my2;
      worldToMapNoBounds(wx2, wy2, mx2, my2);

      // Compute the direction and perpendicular vector for this waypoint pair
      Eigen::Vector2f start(wx1, wy1);
      Eigen::Vector2f end(wx2, wy2);
      Eigen::Vector2f direction = end - start;
      Eigen::Vector2f perpendicular(direction.y(), -direction.x());
      perpendicular.normalize();

      // Determine the bounding box for this waypoint pair
      int local_min_i = std::max(min_i, static_cast<int>(std::min(static_cast<int>(mx1), mx2))-margin_);
      int local_min_j = std::max(min_j, static_cast<int>(std::min(static_cast<int>(my1), my2))-margin_);
      int local_max_i = std::min(max_i, static_cast<int>(std::max(static_cast<int>(mx1), mx2))+margin_);
      int local_max_j = std::min(max_j, static_cast<int>(std::max(static_cast<int>(my1), my2))+margin_);

      // Compute and update costs in the costmap for this waypoint pair
      for (int j = local_min_j; j < local_max_j; ++j) {
        for (int i = local_min_i; i < local_max_i; ++i) {
          // Get the world coordinates of the current cell
          double wx, wy;
          mapToWorld(static_cast<unsigned int>(i), static_cast<unsigned int>(j), wx, wy);
          Eigen::Vector2f point(wx, wy);
          // Extend the line from start to end by a distance
          Eigen::Vector2f extended_start = start - direction.normalized() * 2.0;
          Eigen::Vector2f extended_end = end + direction.normalized() * 2.0;
          // Check if the point lies between the previous and current waypoints
          if ((point - extended_start).dot(direction) >= 0 && (point - extended_end).dot(direction) <= 0)
          {
            // Check if the point is within the static map bounds
            int wi = static_cast<int>(wx / map_resolution_);
            int wj = static_cast<int>(wy / map_resolution_);
            if (wi < 0 || wi >= static_cast<int>(static_map_->info.width) || 
                wj < 0 || wj >= static_cast<int>(static_map_->info.height)) {
              continue;
            }
            int index = wj * static_map_->info.width + wi;
            int cost_static_map = static_cast<int>(static_map_->data[index]);
            // If the cell is in the free space, and is known, add it to the search area
            if(cost_static_map!=-1 && cost_static_map!=100) 
            {
              search_area.push_back(Eigen::Vector2f(wx, wy));
            }
            // If the cell is a wall, check if it is a right wall point
            else if(cost_static_map == 100) {
              Eigen::Vector2f wall_point(wx, wy);
              Eigen::Vector2f point_to_wall = wall_point - start;
              float dot_product = point_to_wall.dot(perpendicular);
              // If the dot product is positive, the point is a right wall point
              if (keep_right_) {
                if (dot_product > 0) {
                  right_wall_points.push_back(wall_point);
                }
              }
              else {
                if (dot_product < 0) {
                  right_wall_points.push_back(wall_point);
                }
              }
            }
          }
        }
      }
      if (search_area.size() == 0 || right_wall_points.size() == 0) {
        return;
      }
    }
    // Compute the distances for each cell in the search area to the right wall
    std::vector<float> distances;
    computeDistancesToWall(search_area, right_wall_points, distances);    

    update_costmap_ = false;
  }
  updateWithAddition(master_grid, min_i, min_j, max_i, max_j);
}

void
LaneKeepingLayer::onFootprintChanged()
{

  RCLCPP_WARN(rclcpp::get_logger(
      "nav2_costmap_2d"), "GradientLayer::onFootprintChanged(): num footprint points: %lu",
    layered_costmap_->getFootprint().size());
}

}  // namespace nav2_costmap_2d

// This is the macro allowing a nav2_costmap_2d::LaneKeepingLayer class
// to be registered in order to be dynamically loadable of base type nav2_costmap_2d::Layer.
// Usually places in the end of cpp-file where the loadable class written.
#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(nav2_costmap_2d::LaneKeepingLayer, nav2_costmap_2d::Layer)