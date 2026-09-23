// Copyright (c) 2026 Kiwibot
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <vector>

#include "nav2_costmap_2d/static_layer.hpp"

namespace nav2_costmap_2d
{

/**
 * @brief nav2_costmap_2d::StaticLayer wrapper exposing the erosion internals
 *
 * computeErosionMasks() only depends on the raw map and on the threshold
 * members, so it can be exercised without a lifecycle node or a live costmap.
 */
class StaticLayerErosionTester : public StaticLayer
{
public:
  void configureErosion(double radius, double ring_thickness, bool enabled)
  {
    erosion_radius_ = radius;
    erosion_ring_thickness_ = ring_thickness;
    erosion_enabled_ = enabled;
    lethal_threshold_ = 100;
    unknown_cost_value_ = 255;
  }

  /// Cache the drivable region only, without eroding any window.
  void buildOnly(const nav_msgs::msg::OccupancyGrid & map)
  {
    if (map.info.resolution > 0.0 && map.info.width > 0 && map.info.height > 0) {
      resizeMap(map.info.width, map.info.height, map.info.resolution, 0.0, 0.0);
    }
    buildRawFreeMask(map);
  }

  /// Erode a window covering the whole fixture, as a costmap over it would.
  void erodeWholeMap(const nav_msgs::msg::OccupancyGrid & map)
  {
    buildOnly(map);
    updateErosionWindow(
      0, 0, static_cast<int>(map.info.width) - 1, static_cast<int>(map.info.height) - 1);
  }

  using StaticLayer::updateErosionWindow;

  bool isActive() const {return erosion_active_;}
  bool hasDrivableRegion() const {return !erosion_raw_free_.empty();}
  bool hasWindow() const {return erosion_window_w_ > 0 && erosion_window_h_ > 0;}

  bool isFree(unsigned int mx, unsigned int my, unsigned int) const
  {
    const size_t index = windowIndex(mx, my);
    return index != kNoCell && erosion_window_free_[index];
  }
  bool isRing(unsigned int mx, unsigned int my, unsigned int) const
  {
    const size_t index = windowIndex(mx, my);
    return index != kNoCell && erosion_window_ring_[index];
  }

private:
  static constexpr size_t kNoCell = static_cast<size_t>(-1);
  size_t windowIndex(unsigned int mx, unsigned int my) const
  {
    const int lx = static_cast<int>(mx) - erosion_window_x_;
    const int ly = static_cast<int>(my) - erosion_window_y_;
    if (lx < 0 || ly < 0 || lx >= erosion_window_w_ || ly >= erosion_window_h_) {
      return kNoCell;
    }
    return static_cast<size_t>(ly) * erosion_window_w_ + lx;
  }
};

namespace
{

constexpr unsigned int kWidth = 41;
constexpr unsigned int kHeight = 41;
constexpr double kResolution = 0.1;

// Corridor occupying rows [17, 23] and columns [8, 32], wrapped in a one cell
// lethal ring, with unknown space everywhere else: the same structure the real
// segmapping has (4.8% free, a thin lethal outline, the rest unknown).
nav_msgs::msg::OccupancyGrid makeCorridorMap(double resolution = kResolution)
{
  nav_msgs::msg::OccupancyGrid map;
  map.info.width = kWidth;
  map.info.height = kHeight;
  map.info.resolution = resolution;
  map.data.assign(static_cast<size_t>(kWidth) * kHeight, -1);

  auto at = [&map](unsigned int mx, unsigned int my) -> int8_t & {
      return map.data[static_cast<size_t>(my) * kWidth + mx];
    };

  for (unsigned int my = 16; my <= 24; ++my) {
    for (unsigned int mx = 7; mx <= 33; ++mx) {
      at(mx, my) = 100;
    }
  }
  for (unsigned int my = 17; my <= 23; ++my) {
    for (unsigned int mx = 8; mx <= 32; ++mx) {
      at(mx, my) = 0;
    }
  }
  return map;
}

}  // namespace

TEST(StaticLayerErosion, NoMasksWhenRadiusIsZero)
{
  StaticLayerErosionTester layer;
  layer.configureErosion(0.0, 0.2, true);
  layer.erodeWholeMap(makeCorridorMap());

  EXPECT_FALSE(layer.hasDrivableRegion());
  EXPECT_FALSE(layer.isActive());
}

TEST(StaticLayerErosion, NoMasksWhenResolutionIsInvalid)
{
  StaticLayerErosionTester layer;
  layer.configureErosion(0.3, 0.2, true);
  layer.buildOnly(makeCorridorMap(0.0));

  EXPECT_FALSE(layer.hasDrivableRegion());
  EXPECT_FALSE(layer.isActive());
}

TEST(StaticLayerErosion, GrowsDrivableRegionByTheConfiguredRadius)
{
  StaticLayerErosionTester layer;
  layer.configureErosion(0.3, 0.2, true);  // 3 cells of growth, 2 cells of ring
  layer.erodeWholeMap(makeCorridorMap());

  ASSERT_TRUE(layer.hasDrivableRegion());

  // Corridor rows [17, 23] grow to [14, 26] at mid corridor.
  for (unsigned int my = 14; my <= 26; ++my) {
    EXPECT_TRUE(layer.isFree(20, my, kWidth)) << "row " << my << " should be free";
  }
  EXPECT_FALSE(layer.isFree(20, 13, kWidth));
  EXPECT_FALSE(layer.isFree(20, 27, kWidth));

  // Corridor columns [8, 32] grow to [5, 35].
  for (unsigned int mx = 5; mx <= 35; ++mx) {
    EXPECT_TRUE(layer.isFree(mx, 20, kWidth)) << "col " << mx << " should be free";
  }
  EXPECT_FALSE(layer.isFree(4, 20, kWidth));
  EXPECT_FALSE(layer.isFree(36, 20, kWidth));
}

TEST(StaticLayerErosion, RedrawsTheBoundaryOutsideTheWidenedRegion)
{
  StaticLayerErosionTester layer;
  layer.configureErosion(0.3, 0.2, true);
  layer.erodeWholeMap(makeCorridorMap());

  // Two cells of ring immediately outside the widened corridor, nothing beyond.
  EXPECT_TRUE(layer.isRing(20, 12, kWidth));
  EXPECT_TRUE(layer.isRing(20, 13, kWidth));
  EXPECT_TRUE(layer.isRing(20, 27, kWidth));
  EXPECT_TRUE(layer.isRing(20, 28, kWidth));
  EXPECT_FALSE(layer.isRing(20, 11, kWidth));
  EXPECT_FALSE(layer.isRing(20, 29, kWidth));

  // A cell is never both free and boundary.
  for (unsigned int my = 0; my < kHeight; ++my) {
    for (unsigned int mx = 0; mx < kWidth; ++mx) {
      EXPECT_FALSE(layer.isFree(mx, my, kWidth) && layer.isRing(mx, my, kWidth))
        << "cell " << mx << "," << my << " is both free and boundary";
    }
  }
}

TEST(StaticLayerErosion, WidenedRegionStaysFullyEnclosed)
{
  StaticLayerErosionTester layer;
  layer.configureErosion(0.3, 0.2, true);
  layer.erodeWholeMap(makeCorridorMap());

  // The safety property of shifting the wall instead of dissolving it: no cell
  // of the widened corridor may touch un-eroded space directly, not even
  // diagonally, or the robot could plan straight through the gap.
  for (unsigned int my = 0; my < kHeight; ++my) {
    for (unsigned int mx = 0; mx < kWidth; ++mx) {
      if (!layer.isFree(mx, my, kWidth)) {
        continue;
      }
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          const int nx = static_cast<int>(mx) + dx;
          const int ny = static_cast<int>(my) + dy;
          if ((dx == 0 && dy == 0) || nx < 0 || ny < 0 ||
            nx >= static_cast<int>(kWidth) || ny >= static_cast<int>(kHeight))
          {
            continue;
          }
          const auto ux = static_cast<unsigned int>(nx);
          const auto uy = static_cast<unsigned int>(ny);
          EXPECT_TRUE(layer.isFree(ux, uy, kWidth) || layer.isRing(ux, uy, kWidth))
            << "leak from " << mx << "," << my << " to " << nx << "," << ny;
        }
      }
    }
  }
}

TEST(StaticLayerErosion, UnknownCellsDoNotSeedGrowth)
{
  // interpretValue() turns unknown into FREE_SPACE when track_unknown_space is
  // false, as the local costmap has it. If the erosion ran on interpreted costs
  // the whole map outside the corridor would count as drivable and the boundary
  // would be dissolved instead of shifted.
  StaticLayerErosionTester layer;
  layer.configureErosion(0.3, 0.2, true);

  nav_msgs::msg::OccupancyGrid map;
  map.info.width = kWidth;
  map.info.height = kHeight;
  map.info.resolution = kResolution;
  map.data.assign(static_cast<size_t>(kWidth) * kHeight, -1);
  layer.erodeWholeMap(map);

  ASSERT_TRUE(layer.hasDrivableRegion());
  for (unsigned int my = 0; my < kHeight; ++my) {
    for (unsigned int mx = 0; mx < kWidth; ++mx) {
      EXPECT_FALSE(layer.isFree(mx, my, kWidth));
      EXPECT_FALSE(layer.isRing(mx, my, kWidth));
    }
  }

  // Far-away unknown space is untouched when a corridor does exist.
  StaticLayerErosionTester corridor_layer;
  corridor_layer.configureErosion(0.3, 0.2, true);
  corridor_layer.erodeWholeMap(makeCorridorMap());
  EXPECT_FALSE(corridor_layer.isFree(0, 0, kWidth));
  EXPECT_FALSE(corridor_layer.isRing(0, 0, kWidth));
}

TEST(StaticLayerErosion, ToggleOffCostsNothingPerCycle)
{
  // With the erosion windowed, nothing is precomputed beyond the drivable
  // region: when the toggle is off no window is built at all, so a way that
  // does not ask for the erosion pays no per cycle cost.
  StaticLayerErosionTester layer;
  layer.configureErosion(0.3, 0.2, false);
  layer.erodeWholeMap(makeCorridorMap());

  EXPECT_TRUE(layer.hasDrivableRegion());
  EXPECT_FALSE(layer.isActive());
  EXPECT_FALSE(layer.hasWindow());
}

TEST(StaticLayerErosion, WindowOnlyCoversTheRequestedRectangle)
{
  // The point of the windowed erosion: a costmap window asks for its own slice
  // of the map, not the whole thing.
  StaticLayerErosionTester layer;
  layer.configureErosion(0.3, 0.2, true);
  layer.buildOnly(makeCorridorMap());
  layer.updateErosionWindow(18, 18, 22, 22);

  ASSERT_TRUE(layer.hasWindow());
  // Inside the requested rectangle the corridor is eroded as usual.
  EXPECT_TRUE(layer.isFree(20, 20, kWidth));
  // Far outside it nothing was computed, so the layer falls back to the raw map.
  EXPECT_FALSE(layer.isFree(20, 2, kWidth));
  EXPECT_FALSE(layer.isRing(20, 2, kWidth));
}

TEST(StaticLayerErosion, RadiusIsNotShrunkByFloatingPointError)
{
  // radius / resolution is inexact in binary floating point (0.3 / 0.1 is
  // 2.9999999999999996), which would silently cost one cell of growth.
  StaticLayerErosionTester layer;
  layer.configureErosion(0.3, 0.2, true);
  layer.erodeWholeMap(makeCorridorMap());

  EXPECT_TRUE(layer.isFree(20, 14, kWidth)) << "third cell of growth was dropped";
}

TEST(StaticLayerErosion, ProductionRadiusReachesExactlyTenCells)
{
  // Regression test for the float32 info.resolution: 1.0 / 0.1f is 9.99999985,
  // not 10, so a tolerance that does not scale with the radius drops the whole
  // outermost axis aligned ring of the disc. Single free cell, so the expected
  // mask is exactly the disc of radius 10 around it.
  nav_msgs::msg::OccupancyGrid map;
  map.info.width = kWidth;
  map.info.height = kHeight;
  map.info.resolution = kResolution;
  map.data.assign(static_cast<size_t>(kWidth) * kHeight, -1);
  map.data[static_cast<size_t>(20) * kWidth + 20] = 0;

  StaticLayerErosionTester layer;
  layer.configureErosion(1.0, 0.2, true);
  layer.erodeWholeMap(map);

  ASSERT_TRUE(layer.hasDrivableRegion());
  EXPECT_TRUE(layer.isFree(30, 20, kWidth)) << "+x cell at exactly 10 cells dropped";
  EXPECT_TRUE(layer.isFree(10, 20, kWidth)) << "-x cell at exactly 10 cells dropped";
  EXPECT_TRUE(layer.isFree(20, 30, kWidth)) << "+y cell at exactly 10 cells dropped";
  EXPECT_TRUE(layer.isFree(20, 10, kWidth)) << "-y cell at exactly 10 cells dropped";
  EXPECT_FALSE(layer.isFree(31, 20, kWidth)) << "grew past the configured radius";

  // And the mask is exactly the disc: no more, no less.
  for (unsigned int my = 0; my < kHeight; ++my) {
    for (unsigned int mx = 0; mx < kWidth; ++mx) {
      const int dx = static_cast<int>(mx) - 20;
      const int dy = static_cast<int>(my) - 20;
      EXPECT_EQ(dx * dx + dy * dy <= 100, layer.isFree(mx, my, kWidth))
        << "cell " << mx << "," << my;
    }
  }
}

}  // namespace nav2_costmap_2d
