//    *************************** LiveVisionKit ****************************
//    Copyright (C) 2022  Sebastian Di Marco (crowsinc.dev@gmail.com)
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <https://www.gnu.org/licenses/>.
// 	  **********************************************************************

#pragma once

#include "VideoFilter.hpp"
#include "Vision/FrameTracker.hpp"
#include "Vision/PathSmoother.hpp"
#include "Utility/Configurable.hpp"
#include "Data/StreamBuffer.hpp"
#include "Math/WarpMesh.hpp"

#include <mutex>
#include <atomic>
#include <shared_mutex>
#include <string>

namespace lvk
{
    struct StabilizationFilterSettings : public FrameTrackerSettings, public PathSmootherSettings
    {
        cv::Size motion_resolution = {2, 2};

        cv::Scalar background_colour = {255,0,255};
        bool crop_to_stable_region = false;
        bool stabilize_output = true;

        // Quality Assurance
        float min_scene_quality = 0.8f;
        float min_tracking_quality = 0.3f;
    };

    class StabilizationFilter final : public VideoFilter, public Configurable<StabilizationFilterSettings>
    {
    public:
        explicit StabilizationFilter(const StabilizationFilterSettings& settings = {});

        void configure(const StabilizationFilterSettings& settings) override;
        void restart();
        bool ready() const;
        void reset_context();
        void draw_trackers();
        void draw_motion_mesh();
        size_t frame_delay() const;
        cv::Rect stable_region() const;

    private:
        void filter(VideoFrame&& input, VideoFrame& output) override;

        // Performance optimization: Pre-allocated working buffers
        VideoFrame m_GrayBuffer;
        VideoFrame m_TempBuffer;
        bool m_BuffersInitialized = false;

        void ensureBuffersAllocated(const cv::Size& frameSize);
        bool convertToGray(const VideoFrame& input, VideoFrame& output, std::string& error) const;

        // Processing context for cleaner method decomposition
        struct ProcessingContext {
            VideoFrame trackingFrame;
            MotionVector motion{WarpMesh::MinimumSize};
            float trackingQuality;
            bool shouldStabilize;
        };

        // Method decomposition for better maintainability
        bool prepareFrame(const VideoFrame& input, ProcessingContext& context, std::string& error);
        bool trackMotion(const VideoFrame& input, ProcessingContext& context, std::string& error);
        void stabilizeFrame(ProcessingContext& context, VideoFrame& output);
        void processFrame(ProcessingContext& context, VideoFrame&& input, VideoFrame& output);
        void updateQualityMetrics(float trackingQuality);
        MotionVector applyQualityControl(const MotionVector& motion) const;
        void processStabilizedOutput(const MotionVector& correctedMotion, VideoFrame& output);
        void handlePassthroughMode(VideoFrame&& input, VideoFrame& output);

        // Core processing components
        FrameTracker m_FrameTracker;
        PathSmoother m_PathSmoother;
        StreamBuffer<Frame> m_FrameQueue{1};
        VideoFrame m_WarpFrame, m_TrackingFrame;
        WarpMesh m_NullMotion{WarpMesh::MinimumSize};

        // Thread safety: Atomic state variables
        std::atomic<float> m_SceneQuality{1.0f};
        std::atomic<float> m_TrustFactor{1.0f};
        std::atomic<bool> m_IsProcessing{false};

        // Thread safety: Synchronization primitives
        mutable std::shared_mutex m_configMutex;
        mutable std::mutex m_processingMutex;
    };
}