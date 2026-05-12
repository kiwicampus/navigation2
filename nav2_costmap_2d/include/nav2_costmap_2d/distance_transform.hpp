// Copyright (c) 2026
// SPDX-License-Identifier: Apache-2.0
//
// 2D squared Euclidean distance transform (Felzenszwalb & Huttenlocher, separable)
// on a row-major float matrix. Seeds are 0; free space is DT_INF.

#ifndef NAV2_COSTMAP_2D__DISTANCE_TRANSFORM_HPP_
#define NAV2_COSTMAP_2D__DISTANCE_TRANSFORM_HPP_

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace nav2_costmap_2d
{

using MatrixXfRM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

namespace DistanceTransform
{

static constexpr float DT_INF = 1e20f;

namespace detail
{
inline void dt1d(const float * f, float * d, int n, std::vector<int> & v, std::vector<float> & z)
{
  int k = 0;
  v[0] = 0;
  z[0] = -DT_INF;
  z[1] = DT_INF;
  for (int q = 1; q < n; ++q) {
    float s = ((f[q] + static_cast<float>(q * q)) - (f[v[k]] + static_cast<float>(v[k] * v[k]))) /
      (2.0f * static_cast<float>(q - v[k]));
    while (s <= z[k]) {
      --k;
      s = ((f[q] + static_cast<float>(q * q)) - (f[v[k]] + static_cast<float>(v[k] * v[k]))) /
        (2.0f * static_cast<float>(q - v[k]));
    }
    ++k;
    v[k] = q;
    z[k] = s;
    z[k + 1] = DT_INF;
  }
  k = 0;
  for (int q = 0; q < n; ++q) {
    while (z[k + 1] < static_cast<float>(q)) {
      ++k;
    }
    const float dq = static_cast<float>(q - v[k]);
    d[q] = f[v[k]] + dq * dq;
  }
}
}  // namespace detail

/** @brief In-place EDT: input seeds 0, non-seed DT_INF; output Euclidean distance in grid units. */
inline void distanceTransform2D(MatrixXfRM & m, int height, int width)
{
  if (height <= 0 || width <= 0) {
    return;
  }

  std::vector<float> buf(std::max(height, width));
  std::vector<float> out(std::max(height, width));
  std::vector<int> v(std::max(height, width));
  std::vector<float> z(static_cast<size_t>(std::max(height, width)) + 1u);

  // Rows: transform along x (columns) for each row index (y).
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      const float val = m(r, c);
      buf[static_cast<size_t>(c)] = (val < 1e-6f) ? 0.0f : DT_INF;
    }
    detail::dt1d(buf.data(), out.data(), width, v, z);
    for (int c = 0; c < width; ++c) {
      m(r, c) = out[static_cast<size_t>(c)];
    }
  }

  // Columns: transform along y (rows) for each column index (x).
  for (int c = 0; c < width; ++c) {
    for (int r = 0; r < height; ++r) {
      buf[static_cast<size_t>(r)] = m(r, c);
    }
    detail::dt1d(buf.data(), out.data(), height, v, z);
    for (int r = 0; r < height; ++r) {
      m(r, c) = out[static_cast<size_t>(r)];
    }
  }

  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      float v_sq = m(r, c);
      if (v_sq >= DT_INF * 0.25f) {
        m(r, c) = DT_INF;
      } else {
        m(r, c) = std::sqrt(std::max(0.0f, v_sq));
      }
    }
  }
}

}  // namespace DistanceTransform

static constexpr int COST_LUT_PRECISION = 100;

}  // namespace nav2_costmap_2d

#endif  // NAV2_COSTMAP_2D__DISTANCE_TRANSFORM_HPP_
