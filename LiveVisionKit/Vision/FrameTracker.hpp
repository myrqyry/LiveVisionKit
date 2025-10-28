//     *************************** LiveVisionKit ****************************
//     Copyright (C) 2022  Sebastian Di Marco (crowsinc.dev@gmail.com)
//
//     This program is free software: you can redistribute it and/or modify
//     it under the terms of the GNU General Public License as published by
//     the Free Software Foundation, either version 3 of the License, or
//     (at your option) any later version.
//
//     This program is distributed in the hope that it will be useful,
//     but WITHOUT ANY WARRANTY; without even the implied warranty of
//     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//     GNU General Public License for more details.
//
//     You should have received a copy of the GNU General Public License
//     along with this program.  If not, see <https://www.gnu.org/licenses/>.
//     **********************************************************************

#pragma once

#include <opencv2/opencv.hpp>

#include "Utility/Configurable.hpp"
#include "FeatureDetector.hpp"
#include "Math/WarpMesh.hpp"
#include "Eigen/Geometry"
#include "Eigen/Sparse"
#include <vector>
#include <algorithm>

namespace lvk
{

struct FrameTrackerMemoryPool {
    std::vector<cv::Point2f> tracked_points_pool, matched_points_pool;
    std::vector<uint8_t> match_status_pool, inlier_status_pool;
    std::vector<cv::KeyPoint> keypoint_pool;
    std::vector<Eigen::Triplet<float>> dynamic_constraints_pool;

    size_t current_capacity = 0;

    void ensure_capacity(size_t required_capacity) {
        if (required_capacity > current_capacity) {
            const size_t new_capacity = std::max(required_capacity, current_capacity * 2);

            tracked_points_pool.reserve(new_capacity);
            matched_points_pool.reserve(new_capacity);
            match_status_pool.reserve(new_capacity);
            inlier_status_pool.reserve(new_capacity);
            keypoint_pool.reserve(new_capacity);
            dynamic_constraints_pool.reserve(new_capacity * 8);

            current_capacity = new_capacity;
        }

        tracked_points_pool.clear();
        matched_points_pool.clear();
        match_status_pool.clear();
        inlier_status_pool.clear();
        keypoint_pool.clear();
        dynamic_constraints_pool.clear();
    }
};

struct OpticalFlowConfig {
    cv::Size window_size{11, 11};
    int pyramid_levels = 3;
    int max_iterations = 5;
    double termination_epsilon = 0.01;
    double homography_distribution_threshold = 0.6;

    bool is_valid() const {
        return window_size.width > 0 && window_size.height > 0 &&
               pyramid_levels > 0 && max_iterations > 0 &&
               termination_epsilon > 0 &&
               homography_distribution_threshold > 0 &&
               homography_distribution_threshold <= 1.0;
    }
};

    struct FrameTrackerSettings : public FeatureDetectorSettings
    {
        cv::Size motion_resolution = {16, 16};

        bool track_local_motions = true;
        float temporal_smoothing = 1.0f;
        float local_smoothing = 20.0f;

        size_t min_motion_samples = 75;
        float acceptance_threshold = 8.0f;
        float uniformity_threshold = 0.20f;
        OpticalFlowConfig optical_flow;
    };

	class FrameTracker final : public Configurable<FrameTrackerSettings>
	{
	public:

		explicit FrameTracker(const FrameTrackerSettings& settings = {});

        void configure(const FrameTrackerSettings& settings) override;

		std::optional<WarpMesh> track(const cv::UMat& next_frame);

		void restart();

        float tracking_stability() const noexcept;

        const cv::Size& motion_resolution() const noexcept;

        const cv::Size& tracking_resolution() const noexcept;

		const std::vector<cv::KeyPoint>& features() const;

        void draw_trackers(cv::UMat& dst, const cv::Scalar& color, const int size = 10, const int thickness = 3) const;

    private:

        int generate_mesh_constraints(
            const cv::Rect2f& region,
            const cv::Size& mesh_size,
            std::vector<Eigen::Triplet<float>>& constraints
        );

        void estimate_local_motions(
            WarpMesh& motion_mesh,
            const cv::Rect2f& region,
            const std::vector<cv::Point2f>& tracked_points,
            const std::vector<cv::Point2f>& matched_points,
            std::vector<uint8_t>& inlier_status
        );

        void estimate_global_motion(
            WarpMesh& motion_mesh,
            const bool homography,
            const cv::Rect2f& region,
            const std::vector<cv::Point2f>& tracked_points,
            const std::vector<cv::Point2f>& matched_points,
            std::vector<uint8_t>& inlier_status
        );

    private:
        bool m_FrameInitialized = false;
        cv::UMat m_PreviousFrame, m_CurrentFrame;

        FeatureDetector m_FeatureDetector;
        std::vector<cv::KeyPoint> m_TrackedFeatures;

        cv::Rect2f m_TrackingRegion;
        float m_TrackingStability = 0;
        cv::Ptr<cv::SparsePyrLKOpticalFlow> m_OpticalTracker = nullptr;

        Eigen::VectorXf m_OptimizedMesh;
        std::vector<Eigen::Triplet<float>> m_StaticMeshConstraints;
        int m_StaticConstraintCount = 0;

        FrameTrackerMemoryPool m_MemoryPool;
        OpticalFlowConfig m_OpticalFlowConfig;

        std::vector<cv::Point2f>& tracked_points() { return m_MemoryPool.tracked_points_pool; }
        std::vector<cv::Point2f>& matched_points() { return m_MemoryPool.matched_points_pool; }
        std::vector<uint8_t>& match_status() { return m_MemoryPool.match_status_pool; }
        std::vector<uint8_t>& inlier_status() { return m_MemoryPool.inlier_status_pool; }
	};
}
