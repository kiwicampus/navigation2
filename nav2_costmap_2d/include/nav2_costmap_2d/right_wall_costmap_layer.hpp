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

class RightWallCostmapLayer : public CostmapLayer
{
public:
  RightWallCostmapLayer();
  virtual ~RightWallCostmapLayer();

  virtual void onInitialize();
  virtual void updateBounds(double robot_x, double robot_y, double robot_yaw, double *min_x,
                            double *min_y, double *max_x, double *max_y);
  virtual void updateCosts(nav2_costmap_2d::Costmap2D &master_grid, int min_i, int min_j,
                           int max_i, int max_j);
  // virtual bool isClearable() override { return false; }
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
  void computeCostmap();
  void computeDistancesToWall(const std::vector<Eigen::Vector2f>& search_area,
                              const std::vector<Eigen::Vector2f>& right_wall_points,
                              std::vector<float>& distances,
                              double* min_x, double* min_y, double* max_x, double* max_y);
  unsigned char computeCost(float distance);
  void setCost(unsigned int mx, unsigned int my, unsigned char cost);

  double last_min_x_, last_min_y_, last_max_x_, last_max_y_;

  // Variables
  nav_msgs::msg::Path::SharedPtr global_path_;
  geometry_msgs::msg::PoseStamped robot_pose_;
  nav_msgs::msg::OccupancyGrid::SharedPtr static_map_;

  // Subscriptions
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;

  // Publishers
  // Image publisher for debugging
  // rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;


  // Parameters
  double roi_size_;            // Region of interest size in meters
  double max_distance_;        // Maximum distance to consider in meters
  double lookahead_distance_;  // Lookahead distance to find next goal point in meters
  double robot_x_, robot_y_, robot_yaw_;
  double prev_goal_x_, prev_goal_y_, goal_x_, goal_y_;
  std::string map_topic_;
  bool current_;
  bool was_reset_;
  std::string global_frame_;
  bool rolling_window_;
  // Size of gradient in cells
  int GRADIENT_SIZE = 20;
  // Step of increasing cost per one cell in gradient
  int GRADIENT_FACTOR = 10;
  bool need_recalculation_;
  int num_max_files_ = 5;
  int num_files_ = 0;
  int path_index_ = 0;
  double map_resolution_ = 0.1;
  bool is_local_=false;


  // Costmap buffer
  nav2_costmap_2d::Costmap2D costmap_buffer_;

  // Mutex for thread safety
  std::mutex data_mutex_;
};

}  // namespace nav2_costmap_2d

#endif  // RIGHT_WALL_COSTMAP_LAYER_HPP_
