#ifndef RIGHT_WALL_COSTMAP_LAYER_HPP_
#define RIGHT_WALL_COSTMAP_LAYER_HPP_

#include "nav2_costmap_2d/layer.hpp"
#include "nav2_costmap_2d/layered_costmap.hpp"
#include "nav2_costmap_2d/costmap_layer.hpp"
#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "Eigen/Core"
#include "Eigen/Dense"
#include "nanoflann.hpp"

#include <mutex>
#include <vector>

namespace nav2_costmap_2d
{

/**
 * @brief Wall struct
 * Struct to store the wall 2D points and provide methods to interact with the nanoflann library
 */
struct Wall
{
    std::vector<Eigen::Vector2f> pts;

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
        return pts[idx][dim];
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }
};

/**
 * @brief RightWallCostmapLayer class
 * @details Class that implements the right wall costmap layer
 * The layer is used to create a costmap based on the distance to the right wall
 * Most of the layer logic is implemented in the updateCosts method, which is called at each iteration
 * It uses the global path to know in which direction the robot should drive and identify the right wall in the static map
 * It uses the nanoflann library to perform kd-tree searches and find the nearest wall point for each cell in the free space.
 * We compute the cost for eachcell based on the computeCost method
 */
class RightWallCostmapLayer : public CostmapLayer
{
public:
  RightWallCostmapLayer();
  virtual ~RightWallCostmapLayer();

  virtual void onInitialize();
  /**
   * @brief Update the bounds of the costmap
   * The method is called to ask the plugin: which area of costmap it needs to update.
   * Inside this method window bounds are re-calculated
   * and updated independently on its value.
   * @param robot_x x position of the robot
   * @param robot_y y position of the robot
   * @param robot_yaw yaw of the robot
   * @param min_x pointer to the minimum x value of the window bounds
   * @param min_y pointer to the minimum y value of the window bounds
   * @param max_x pointer to the maximum x value of the window bounds
   * @param max_y pointer to the maximum y value of the window bounds
   */
  virtual void updateBounds(double robot_x, double robot_y, double robot_yaw, double *min_x,
                            double *min_y, double *max_x, double *max_y);

  /**
   * @brief Update the costs of the costmap
   * The method is called to update the costmap within its window bounds.
   * @param master_grid costmap to update
   * @param min_i minimum index in the i direction
   * @param min_j minimum index in the j direction
   * @param max_i maximum index in the i direction
   * @param max_j maximum index in the j direction
   */
  virtual void updateCosts(nav2_costmap_2d::Costmap2D &master_grid, int min_i, int min_j,
                           int max_i, int max_j);
  // virtual bool isClearable() override { return false; }
  rcl_interfaces::msg::SetParametersResult dynamicParametersCallback(std::vector<rclcpp::Parameter> parameters);

  virtual void reset();
  virtual void onFootprintChanged();
  virtual bool isClearable() { return false; }
  virtual void activate();
  virtual void deactivate();

private:
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
  void pathCallback(const nav_msgs::msg::Path::SharedPtr msg);
  void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
  void getParameters();
  /**
   * @brief Compute the distances to the wall for each cell in the free space
   * @param search_area points of the free space
   * @param right_wall_points points of the right wall
   * @param distances vector to store the distances to the wall for each cell in the free space
   */
  void computeDistancesToWall(const std::vector<Eigen::Vector2f>& search_area,
                              const std::vector<Eigen::Vector2f>& right_wall_points,
                              std::vector<float>& distances);
  /**
   * @brief Compute the cost for a given distance - Currently the cost is computed linearly directly proportional to the distance
   * @param distance distance to the wall
   * @return cost value
   */
  unsigned char computeCost(float distance);
  /**
   * @brief Set the cost for a given cell in the costmap
   * @param mx x index of the cell
   * @param my y index of the cell
   * @param cost cost value to set
   */
  void setCost(unsigned int mx, unsigned int my, unsigned char cost);

  double last_min_x_, last_min_y_, last_max_x_, last_max_y_;

  // Variables
  nav_msgs::msg::Path::SharedPtr global_path_;
  geometry_msgs::msg::PoseStamped robot_pose_;
  nav_msgs::msg::OccupancyGrid::SharedPtr static_map_;
  double robot_x_, robot_y_;
  double prev_goal_x_, prev_goal_y_, goal_x_, goal_y_;
  bool was_reset_;
  int path_index_ = 0;
  bool update_costmap_ = true;
  int margin_;

  // Subscriptions
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;

  // Parameters
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr dyn_params_handler_;

  double max_distance_;        // Maximum distance from the right wall in meters
  double max_cost_, min_cost_, map_resolution_;
  bool rolling_window_;
  std::string map_topic_;
  std::string global_path_topic_;
  std::string global_odom_topic_;
  std::string global_frame_;

  // Costmap buffer
  nav2_costmap_2d::Costmap2D costmap_buffer_;

  // Mutex for thread safety
  std::mutex data_mutex_;
};

}  // namespace nav2_costmap_2d

#endif  // RIGHT_WALL_COSTMAP_LAYER_HPP_