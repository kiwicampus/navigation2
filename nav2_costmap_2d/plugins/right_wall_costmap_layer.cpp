#include "nav2_costmap_2d/right_wall_costmap_layer.hpp"
#include "nav2_costmap_2d/costmap_math.hpp"
#include <algorithm>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>


namespace nav2_costmap_2d
{

RightWallCostmapLayer::RightWallCostmapLayer()
: last_min_x_(-std::numeric_limits<float>::max()),
last_min_y_(-std::numeric_limits<float>::max()),
last_max_x_(std::numeric_limits<float>::max()),
last_max_y_(std::numeric_limits<float>::max())
{
}

RightWallCostmapLayer::~RightWallCostmapLayer()
{
}

void RightWallCostmapLayer::onInitialize()
{
  RCLCPP_INFO(logger_, "RightWallCostmapLayer::onInitialize()");
  current_ = true;
  was_reset_ = false;
  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error{"Failed to lock node"};
  }
  RCLCPP_INFO(logger_, "RightWallCostmapLayer::onInitialize()2");

  declareParameter("enabled", rclcpp::ParameterValue(true));
  declareParameter("roi_size", rclcpp::ParameterValue(10.0));          // in meters
  declareParameter("max_distance", rclcpp::ParameterValue(5.0));       // in meters
  declareParameter("map_topic", rclcpp::ParameterValue("/map"));
  declareParameter("is_local", rclcpp::ParameterValue(false));

  getParameters();
  RCLCPP_INFO(logger_, "RightWallCostmapLayer::onInitialize()3");

  // need_recalculation_ = false;

  // Set up subscriptions

  odom_sub_ = node->create_subscription<nav_msgs::msg::Odometry>(
    "/odometry/global", rclcpp::SensorDataQoS(),
    std::bind(&RightWallCostmapLayer::odomCallback, this, std::placeholders::_1));

  path_sub_ = node->create_subscription<nav_msgs::msg::Path>(
    "/nav2/cartesian_geopath", rclcpp::SystemDefaultsQoS(),
    std::bind(&RightWallCostmapLayer::pathCallback, this, std::placeholders::_1));

  map_sub_ = node->create_subscription<nav_msgs::msg::OccupancyGrid>(
    map_topic_, rclcpp::QoS(10).transient_local().reliable().keep_last(1),
    std::bind(&RightWallCostmapLayer::mapCallback, this, std::placeholders::_1));

  RCLCPP_INFO(logger_, "RightWallCostmapLayer::onInitialize()4");


  enabled_ = true;
  

  global_frame_ = layered_costmap_->getGlobalFrameID();
  rolling_window_ = layered_costmap_->isRolling();
  default_value_ = NO_INFORMATION;

  need_recalculation_ = false;
  matchSize();
  RCLCPP_INFO(logger_, "RightWallCostmapLayer::onInitialize()5");

}

void
RightWallCostmapLayer::activate()
{
}

void
RightWallCostmapLayer::deactivate()
{
  auto node = node_.lock();
}

void RightWallCostmapLayer::reset()
{
  resetMaps();
  current_ = false;
  was_reset_ = true;
}

void RightWallCostmapLayer::getParameters()
{
  auto node = node_.lock();
  node->get_parameter(name_ + "." + "enabled", enabled_);
  node->get_parameter(name_ + "." + "roi_size", roi_size_);
  node->get_parameter(name_ + "." + "max_distance", max_distance_);
  node->get_parameter(name_ + "." + "map_topic", map_topic_);
  node->get_parameter(name_ + "." + "is_local", is_local_);

}

void RightWallCostmapLayer::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  // RCLCPP_INFO(logger_, "Received odometry message");
  std::lock_guard<std::mutex> guard(data_mutex_);
  robot_pose_.header = msg->header;
  robot_pose_.pose = msg->pose.pose;
  robot_x_ = robot_pose_.pose.position.x;
  robot_y_ = robot_pose_.pose.position.y;
  // prev_goal_x_ = robot_pose_.pose.position.x;
  // prev_goal_y_ = robot_pose_.pose.position.y;
}

void RightWallCostmapLayer::pathCallback(const nav_msgs::msg::Path::SharedPtr msg)
{
  std::lock_guard<std::mutex> guard(data_mutex_);
  global_path_ = msg;

  // find prev_goal_x_ and prev_goal_y_ from the path at index 0 and goal_x_, goal_y_ from the path at index 1
  path_index_ = 0;
  if(path_index_ + 1 >= static_cast<int>(global_path_->poses.size())) {
    RCLCPP_WARN(logger_, "RightWallCostmapLayer::pathCallback() path_index_ is out of bounds");
    return;
  }
  prev_goal_x_ = global_path_->poses[path_index_].pose.position.x;
  prev_goal_y_ = global_path_->poses[path_index_].pose.position.y;
  goal_x_ = global_path_->poses[path_index_+1].pose.position.x;
  goal_y_ = global_path_->poses[path_index_+1].pose.position.y;
  // Print path
  for (int i = 0; i < static_cast<int>(global_path_->poses.size()); i++) {
    RCLCPP_WARN(logger_, "RightWallCostmapLayer::pathCallback() path[%d]: %f, %f", i, global_path_->poses[i].pose.position.x, global_path_->poses[i].pose.position.y);
  }
  // computeCostmap();
}

void RightWallCostmapLayer::mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
{
  std::lock_guard<std::mutex> guard(data_mutex_);
  RCLCPP_WARN(logger_, "Received map message");
  static_map_ = msg;
  // processMap();
}

// unsigned char RightWallCostmapLayer::computeCost(float distance)
// {

//     const float max_distance = 2.0f; // Maximum distance to consider
//     const float max_cost = 240.0f;
//     const float min_cost = 1.0f;

//     if (distance > max_distance)
//     {
//         return max_cost;
//     }
//     else
//     {
//         // Linear interpolation between min_cost and max_cost
//         return static_cast<unsigned char>(min_cost + (max_cost - min_cost) * distance / max_distance);
//     }
// }

unsigned char RightWallCostmapLayer::computeCost(float distance)
{
    const float max_distance = 4.0f; // Maximum distance to consider
    const float max_cost = 100.0f;
    const float min_cost = 0.0f;

    if (distance > max_distance)
    {
        return max_cost*0;
    }
    else
    {
        // Exponential interpolation between min_cost and max_cost
        // return static_cast<unsigned char>(min_cost + (max_cost - min_cost) * (1.0-std::exp(-distance / max_distance)));
        return static_cast<unsigned char>(min_cost + (max_cost - min_cost) * distance / max_distance)*0;

    }
}


void RightWallCostmapLayer::setCost(unsigned int mx, unsigned int my, unsigned char cost)
{
    // Check bounds
    if (mx < getSizeInCellsX() && my < getSizeInCellsY())
    {
        // unsigned char* costmap = getCharMap();
        unsigned int index = getIndex(mx, my);
        costmap_[index] = cost;
    }
    else
    {
        RCLCPP_WARN(logger_, "RightWallCostmapLayer::setCost() index out of bounds");
    }
}

void RightWallCostmapLayer::computeDistancesToWall(
    const std::vector<Eigen::Vector2f>& search_area,
    const std::vector<Eigen::Vector2f>& wall_points,
    std::vector<float>& distances)
{
    // Build PointCloud from wall_points
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
        unsigned char cost = computeCost(distances[i]);
        // float mx, my;
        unsigned int mx, my;
        // worldToMapContinuous(search_area[i].x(), search_area[i].y(), mx, my);
        // RCLCPP_WARN(logger_, "RightWallCostmapLayer::computeDistancesToWall() mx: %f, my: %f, origin_x: %f, origin_y: %f, size_x: %f, size_y: %f", mx, my, getOriginX(), getOriginY(), getSizeInMetersX(), getSizeInMetersY());
        setCost(static_cast<unsigned int>(mx), static_cast<unsigned int>(my), cost);
        if (!worldToMap(search_area[i].x(), search_area[i].y(), mx, my))
        {

          RCLCPP_WARN(logger_, "Computing map coords failed: %f, %f, limits: %f, %f", search_area[i].x()-robot_x_, search_area[i].y()-robot_y_, getSizeInMetersX(), getSizeInMetersY());
          continue;
        }
        // mx = static_cast<unsigned int>(search_area[i].x()/map_resolution_);
        // my = static_cast<unsigned int>(search_area[i].y()/map_resolution_);
        // RCLCPP_WARN(logger_, "RightWallCostmapLayer::computeDistancesToWall() mx: %d, my: %d, cost: %d", mx, my, cost);
        setCost(mx, my, cost);
        // touch(search_area[i].x(), search_area[i].y(),min_x, min_y, max_x, max_y);
    }
}


// The method is called to ask the plugin: which area of costmap it needs to update.
// Inside this method window bounds are re-calculated if need_recalculation_ is true
// and updated independently on its value.
void RightWallCostmapLayer::updateBounds(double robot_x, double robot_y, double /*robot_yaw*/,
                                             double* min_x, double* min_y, double* max_x,
                                             double* max_y)
{
  RCLCPP_WARN(logger_, "RightWallCostmapLayer::updateBounds()");
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
  

  // int min_i = std::max(0, static_cast<int>((prev_goal_x_ - roi_size_/2)/map_resolution_));
  // int min_j = std::max(0, static_cast<int>((prev_goal_y_ - roi_size_/2)/map_resolution_));
  // int max_i = std::min(static_cast<int>(static_map_->info.width), static_cast<int>((prev_goal_x_ + roi_size_/2)/map_resolution_));
  // int max_j = std::min(static_cast<int>(static_map_->info.height), static_cast<int>((prev_goal_y_ + roi_size_/2)/map_resolution_));


  // if(num_files_ < 5) {
  //   num_files_++;
  //   // Open a CSV file to write the data
  //   std::ofstream file("/workspace/scripts/keep_right/walls_data" + std::to_string(num_files_) + ".csv");
  //   if (!file.is_open())
  //   {
  //       RCLCPP_ERROR(logger_, "Failed to open file for writing.");
  //       return;
  //   }
  //   file << "robot_x,robot_y,goal_x,goal_y,wall_x,wall_y,is_right_wall\n";
  //   Eigen::Vector2f robot_position(prev_goal_x_, prev_goal_y_);
  //   Eigen::Vector2f goal_position(goal_x_, goal_y_);
  //   Eigen::Vector2f goal_direction = goal_position - robot_position;
  //   Eigen::Vector2f perpendicular_direction(goal_direction.y(), -goal_direction.x());

  //   std::vector<Eigen::Vector2f> right_wall_points;
  //   std::vector<Eigen::Vector2f> left_wall_points;
  //   std::vector<Eigen::Vector2f> search_area;
  //   int max_value = 0;
  //   int min_value = 1000;
  //   for(int j = min_j; j < max_j; j++) {
  //     for(int i = min_i; i < max_i; i++) {
  //       int index = j * static_map_->info.width + i;
  //       // auto value = static_map_->data[index];
  //       int cost = static_cast<int>(static_map_->data[index]);
  //       if(cost!=-1 && cost!=100) 
  //       {
  //         search_area.push_back(Eigen::Vector2f(i*map_resolution_, j*map_resolution_));
  //       }
  //       if(cost == 100) {
  //         // check if right or left wall
  //         Eigen::Vector2f wall_point(i*map_resolution_, j*map_resolution_);
  //         Eigen::Vector2f point_to_wall = wall_point - robot_position;
  //         float dot_product = point_to_wall.dot(perpendicular_direction);
  //         if (dot_product > 0) {
  //           right_wall_points.push_back(wall_point);
  //           file << prev_goal_x_ << "," << prev_goal_y_ << "," << goal_x_ << "," << goal_y_ << "," << wall_point.x() << "," << wall_point.y() << ",1\n";
  //         } else {
  //           left_wall_points.push_back(wall_point);
  //           file << prev_goal_x_ << "," << prev_goal_y_ << "," << goal_x_ << "," << goal_y_ << "," << wall_point.x() << "," << wall_point.y() << ",0\n";
  //         }

  //       }
  //       // Convert map indices to world coordinates
  //       // Find min and max values
  //       if (cost > max_value) {
  //         max_value = static_map_->data[index];
  //       }
  //       if (cost < min_value) {
  //         min_value = static_map_->data[index];
  //       }
  //     }
  //       // if(static_map_->data[index] ==
  //   }
  //   file.close();

  //   // Print size of the search area and size of the right wall points
  //   // RCLCPP_WARN(logger_, "RightWallCostmapLayer::updateBounds() search_area size: %lu", search_area.size());
  //   // RCLCPP_WARN(logger_, "RightWallCostmapLayer::updateBounds() right_wall_points size: %lu", right_wall_points.size());

  //   // TODO: Use a KDTree to compute the distances to the right wall for all the poitns in the search area
  //   std::vector<float> distances;
  //   computeDistancesToWall(search_area, right_wall_points, distances, min_x, min_y, max_x, max_y);
    
  //   // Save distances to file
  //   std::ofstream distance_file("/workspace/scripts/keep_right/distance_data" + std::to_string(num_files_) + ".csv");
  //   if (!distance_file.is_open())
  //   {
  //       RCLCPP_ERROR(logger_, "Failed to open file for writing.");
  //       return;
  //   }
  //   distance_file << "x,y,distance\n";
  //   for (size_t i = 0; i < search_area.size(); i++) {
  //     distance_file << search_area[i].x() << "," << search_area[i].y() << "," << distances[i] << "\n";
  //   }
  //   distance_file.close();

    
  // }
  // else{
  //   num_files_ = 0;
  // }

  // // Print min and max

  current_ = true;
}

// The method is called when costmap recalculation is required.
// It updates the costmap within its window bounds.
// Inside this method the costmap gradient is generated and is writing directly
// to the resulting costmap master_grid without any merging with previous layers.
void RightWallCostmapLayer::updateCosts(nav2_costmap_2d::Costmap2D& master_grid, int min_i,
                                            int min_j, int max_i, int max_j)
{
  // RCLCPP_WARN(logger_, "RightWallCostmapLayer::updateCosts(): min_i: %d, min_j: %d, max_i: %d, max_j: %d", min_i, min_j, max_i, max_j);
  RCLCPP_WARN(logger_, "RightWallCostmapLayer::updateCosts() called");
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  if (!enabled_)
  {
    RCLCPP_WARN(logger_, "RightWallCostmapLayer::updateCosts() called but not enabled");
    return;
  }
  if (!current_ && was_reset_)
  {
    RCLCPP_WARN(logger_, "RightWallCostmapLayer::updateCosts() called but not current and was reset");
    was_reset_ = false;
    current_ = true;
  }
  // if (!costmap_)
  // {
  //   return;
  // }

  if(prev_goal_x_ == 0.0 && prev_goal_y_ == 0.0) {
    return;
  }

  // --------------------- Update costmap values --------------------------
 
  // Update the goal points
  // If global path is not available, return
  if(global_path_ == nullptr) {
    return;
  }
   // Check if the robot has reached the goal point
  double distance_to_goal = distance(robot_x_, robot_y_, goal_x_, goal_y_);
  if(distance_to_goal < 4.0) {
    path_index_++;
  }
  if(path_index_ + 1 >= static_cast<int>(global_path_->poses.size())) {
      path_index_--;
      // return;
  }
  prev_goal_x_ = global_path_->poses[path_index_].pose.position.x;
  prev_goal_y_ = global_path_->poses[path_index_].pose.position.y;
  goal_x_ = global_path_->poses[path_index_+1].pose.position.x;
  goal_y_ = global_path_->poses[path_index_+1].pose.position.y;
  RCLCPP_WARN(logger_, "RightWallCostmapLayer::updateCosts() prev_goal_x_: %f, prev_goal_y_: %f, goal_x_: %f, goal_y_: %f", prev_goal_x_, prev_goal_y_, goal_x_, goal_y_);

  if(num_files_ < 5) {
    num_files_++;
    // Open a CSV file to write the data
    std::ofstream walls_file("/workspace/scripts/keep_right/walls_data" + std::to_string(num_files_) + ".csv");
    std::ofstream space_file("/workspace/scripts/keep_right/search_space" + std::to_string(num_files_) + ".csv");
    
    if (!walls_file.is_open())
    {
        RCLCPP_ERROR(logger_, "Failed to open file for writing.");
        return;
    }
    if (!space_file.is_open())
    {
        RCLCPP_ERROR(logger_, "Failed to open file for writing.");
        return;
    }
    walls_file << "robot_x,robot_y,goal_x,goal_y,wall_x,wall_y,is_right_wall\n";
    space_file << "x,y\n";
    Eigen::Vector2f robot_position(prev_goal_x_, prev_goal_y_);
    Eigen::Vector2f goal_position(goal_x_, goal_y_);
    Eigen::Vector2f goal_direction = goal_position - robot_position;
    Eigen::Vector2f perpendicular_direction(goal_direction.y(), -goal_direction.x());

    std::vector<Eigen::Vector2f> right_wall_points;
    std::vector<Eigen::Vector2f> left_wall_points;
    std::vector<Eigen::Vector2f> search_area;

    int min_i_map = std::max(0, static_cast<int>((prev_goal_x_ - roi_size_/2)/map_resolution_));
    int min_j_map = std::max(0, static_cast<int>((prev_goal_y_ - roi_size_/2)/map_resolution_));
    int max_i_map = std::min(static_cast<int>(static_map_->info.width), static_cast<int>((prev_goal_x_ + roi_size_/2)/map_resolution_));
    int max_j_map = std::min(static_cast<int>(static_map_->info.height), static_cast<int>((prev_goal_y_ + roi_size_/2)/map_resolution_));

    // Compute and update costs in the costmap
    for (int j = min_j; j < max_j; ++j) {
        for (int i = min_i; i < max_i; ++i) {
          // get x, y coordinates of the cell
          double wx, wy;
          // get the world coordinates of the cell
          mapToWorld(static_cast<unsigned int>(i), static_cast<unsigned int>(j), wx, wy);
          int wi = static_cast<int>(wx/map_resolution_);
          int wj = static_cast<int>(wy/map_resolution_);
          // check if the cell is out of bounds
          if (wi < min_i_map || wi >= max_i_map || wj < min_j_map || wj >= max_j_map) {
            continue;
          }
          
          int index = wj * static_map_->info.width + wi;
          int cost = static_cast<int>(static_map_->data[index]);
          if (cost==-1){
            continue;
          }
          // RCLCPP_WARN(logger_, "RightWallCostmapLayer::updateCosts() i: %d, j: %d, wi: %d, wj: %d, wx: %f, wy: %f", i, j, wi, wj, wx, wy);
          if(cost!=-1 && cost!=100) 
          {
            search_area.push_back(Eigen::Vector2f(wx, wy));
            space_file << wx<< "," << wy << "\n";
          }
          if(cost == 100) {
            // check if right or left wall
            // RCLCPP_WARN(logger_, "RightWallCostmapLayer::updateCosts() Found wall at i: %d, j: %d", i, j);
            Eigen::Vector2f wall_point(wx, wy);
            Eigen::Vector2f point_to_wall = wall_point - robot_position;
            float dot_product = point_to_wall.dot(perpendicular_direction);
            if (dot_product > 0) {
              right_wall_points.push_back(wall_point);
              walls_file << prev_goal_x_ << "," << prev_goal_y_ << "," << goal_x_ << "," << goal_y_ << "," << wall_point.x() << "," << wall_point.y() << ",1\n";
            } else {
              left_wall_points.push_back(wall_point);
              walls_file << prev_goal_x_ << "," << prev_goal_y_ << "," << goal_x_ << "," << goal_y_ << "," << wall_point.x() << "," << wall_point.y() << ",0\n";
            }

          }
          else {
            continue;
          }

        }
    }
    walls_file.close();
    space_file.close();
    RCLCPP_WARN(logger_, "RightWallCostmapLayer::updateCosts() search_area size: %lu", search_area.size());
    RCLCPP_WARN(logger_, "RightWallCostmapLayer::updateCosts() right_wall_points size: %lu", right_wall_points.size());
    if (search_area.size() == 0 || right_wall_points.size() == 0) {
      return;
    }
    std::vector<float> distances;
    computeDistancesToWall(search_area, right_wall_points, distances);
    
    // Save distances to file
    std::ofstream distance_file("/workspace/scripts/keep_right/distance_data" + std::to_string(num_files_) + ".csv");
    if (!distance_file.is_open())
    {
        RCLCPP_ERROR(logger_, "Failed to open file for writing.");
        return;
    }
    distance_file << "x,y,distance\n";
    for (size_t i = 0; i < search_area.size(); i++) {
      distance_file << search_area[i].x() << "," << search_area[i].y() << "," << distances[i] << "\n";
    }
    distance_file.close();
  }
  else{
    num_files_ = 0;
  }
  // RCLCPP_WARN(logger_, "RightWallCostmapLayer::updateCosts(): min_i: %d, min_j: %d, max_i: %d, max_j: %d", min_i, min_j, max_i, max_j);

  // Obtain max and min values in costmap_
  // unsigned char* costmap = getCharMap();
  // unsigned char max_cost = 0;
  // unsigned char min_cost = 255;
  // for (int j = min_j; j < max_j; j++)
  // {
  //   for (int i = min_i; i < max_i; i++)
  //   {
  //     unsigned char cost = costmap[getIndex(i, j)];
  //     if (cost > max_cost)
  //     {
  //       max_cost = cost;
  //     }
  //     if (cost < min_cost)
  //     {
  //       min_cost = cost;
  //     }
  //   }
  // }
  // RCLCPP_WARN(logger_, "RightWallCostmapLayer::updateCosts(): min_cost: %d, max_cost: %d", min_cost, max_cost);
  updateWithAddition(master_grid, min_i, min_j, max_i, max_j);

  // // Obtain max and min values in master_grid
  // unsigned char* master_array = master_grid.getCharMap();
  // unsigned char max_master_cost = 0;
  // unsigned char min_master_cost = 255;
  // for (int j = min_j; j < max_j; j++)
  // {
  //   for (int i = min_i; i < max_i; i++)
  //   {
  //     unsigned char cost = master_array[master_grid.getIndex(i, j)];
  //     if (cost > max_master_cost)
  //     {
  //       max_master_cost = cost;
  //     }
  //     if (cost < min_master_cost)
  //     {
  //       min_master_cost = cost;
  //     }
  //   }
  // }
  // RCLCPP_WARN(logger_, "RightWallCostmapLayer::updateCosts(): min_master_cost: %d, max_master_cost: %d", min_master_cost, max_master_cost);

 
  // updateWithMax(master_grid, min_i, min_j, max_i, max_j);
  
  // // master_array - is a direct pointer to the resulting master_grid.
  // // master_grid - is a resulting costmap combined from all layers.
  // // By using this pointer all layers will be overwritten!
  // // To work with costmap layer and merge it with other costmap layers,
  // // please use costmap_ pointer instead (this is pointer to current
  // // costmap layer grid) and then call one of updates methods:
  // // - updateWithAddition()
  // // - updateWithMax()
  // // - updateWithOverwrite()
  // // - updateWithTrueOverwrite()
  // // In this case using master_array pointer is equal to modifying local costmap_
  // // pointer and then calling updateWithTrueOverwrite():
  // unsigned char * master_array = master_grid.getCharMap();
  // unsigned int size_x = master_grid.getSizeInCellsX(), size_y = master_grid.getSizeInCellsY();

  // // {min_i, min_j} - {max_i, max_j} - are update-window coordinates.
  // // These variables are used to update the costmap only within this window
  // // avoiding the updates of whole area.
  // //
  // // Fixing window coordinates with map size if necessary.
  // min_i = std::max(0, min_i);
  // min_j = std::max(0, min_j);
  // max_i = std::min(static_cast<int>(size_x), max_i);
  // max_j = std::min(static_cast<int>(size_y), max_j);

  // // Simply computing one-by-one cost per each cell
  // int gradient_index;
  // for (int j = min_j; j < max_j; j++) {
  //   // Reset gradient_index each time when reaching the end of re-calculated window
  //   // by OY axis.
  //   gradient_index = 0;
  //   for (int i = min_i; i < max_i; i++) {
  //     int index = master_grid.getIndex(i, j);
  //     // setting the gradient cost
  //     unsigned char cost = (LETHAL_OBSTACLE - gradient_index*GRADIENT_FACTOR)%255;
  //     if (gradient_index <= GRADIENT_SIZE) {
  //       gradient_index++;
  //     } else {
  //       gradient_index = 0;
  //     }
  //     master_array[index] = cost;
  //   }
  // }
  
}

// The method is called when footprint was changed.
// Here it just resets need_recalculation_ variable.
void
RightWallCostmapLayer::onFootprintChanged()
{
  need_recalculation_ = true;

  RCLCPP_WARN(rclcpp::get_logger(
      "nav2_costmap_2d"), "GradientLayer::onFootprintChanged(): num footprint points: %lu",
    layered_costmap_->getFootprint().size());
}



}  // namespace nav2_costmap_2d

// This is the macro allowing a nav2_gradient_costmap_plugin::GradientLayer class
// to be registered in order to be dynamically loadable of base type nav2_costmap_2d::Layer.
// Usually places in the end of cpp-file where the loadable class written.
#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(nav2_costmap_2d::RightWallCostmapLayer, nav2_costmap_2d::Layer)