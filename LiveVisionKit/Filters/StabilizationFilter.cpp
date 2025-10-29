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
//    MERCHANTABILITY or FITNESS for a PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <https://www.gnu.org/licenses/>.
// 	  **********************************************************************

#include "StabilizationFilter.hpp"

#include "Directives.hpp"
#include "Functions/Drawing.hpp"
#include "Functions/Extensions.hpp"

namespace lvk
{

//---------------------------------------------------------------------------------------------------------------------

    constexpr float QA_UPDATE_RATE = 0.1f;
    constexpr float QA_BLEND_STEP = 0.05f;

//---------------------------------------------------------------------------------------------------------------------

	StabilizationFilter::StabilizationFilter(const StabilizationFilterSettings& settings)
		: VideoFilter("Stabilization Filter")
	{
		configure(settings);
	}

//---------------------------------------------------------------------------------------------------------------------

    void StabilizationFilter::ensureBuffersAllocated(const cv::Size& frameSize)
    {
        if (!m_BuffersInitialized || m_GrayBuffer.size() != frameSize) {
            m_GrayBuffer.create(frameSize, CV_8UC1);
            m_GrayBuffer.format = VideoFrame::GRAY;
            m_TempBuffer.create(frameSize, CV_8UC3);
            m_BuffersInitialized = true;
        }
    }

//---------------------------------------------------------------------------------------------------------------------

    bool StabilizationFilter::convertToGray(const VideoFrame& input, VideoFrame& output, std::string& errorMsg) const
    {
        try {
            if (input.format == VideoFrame::GRAY) {
                output = input;
                return true;
            }

            if (cv::ocl::useOpenCL()) {
                cv::cvtColor(input, output, cv::COLOR_BGR2GRAY);
                if (output.empty()) {
                    errorMsg = "OpenCV color conversion failed - output is empty";
                    return false;
                }
            } else {
                input.viewAsFormat(output, VideoFrame::GRAY);
                if (output.empty()) {
                    errorMsg = "Format view conversion failed - output is empty";
                    return false;
                }
            }

            output.format = VideoFrame::GRAY;
            return true;

        } catch (const cv::Exception& e) {
            errorMsg = "OpenCV error in color conversion: " + std::string(e.what());
            return false;
        } catch (const std::exception& e) {
            errorMsg = "Standard exception in color conversion: " + std::string(e.what());
            return false;
        } catch (...) {
            errorMsg = "Unknown exception in color conversion";
            return false;
        }
    }

//---------------------------------------------------------------------------------------------------------------------

    void StabilizationFilter::configure(const StabilizationFilterSettings& settings)
    {
        std::unique_lock<std::shared_mutex> lock(m_configMutex);
        std::lock_guard<std::mutex> processingLock(m_processingMutex);
        LVK_ASSERT_01(settings.min_tracking_quality);
        LVK_ASSERT_01(settings.min_scene_quality);

        m_NullMotion.resize(settings.motion_resolution);

        if(m_Settings.stabilize_output && !settings.stabilize_output)
            reset_context();

        m_Settings = settings;

        static_cast<PathSmootherSettings&>(m_Settings).motion_resolution = settings.motion_resolution;
        static_cast<FrameTrackerSettings&>(m_Settings).motion_resolution = settings.motion_resolution;

        m_PathSmoother.configure(m_Settings);
        m_FrameQueue.resize(m_PathSmoother.time_delay() + 1);

        m_FrameTracker.configure(m_Settings);
    }

//---------------------------------------------------------------------------------------------------------------------

    bool StabilizationFilter::prepareFrame(const VideoFrame& input, ProcessingContext& context, std::string& errorMsg)
    {
        // Convert to grayscale for tracking
        std::string conversionError;
        if (!convertToGray(input, context.trackingFrame, conversionError)) {
            errorMsg = "Frame conversion failed: " + conversionError;
            return false;
        }

        // Track motion
        auto motionResult = m_FrameTracker.track(context.trackingFrame);
        if (motionResult.has_value()) {
            context.motion = motionResult.value();
            context.trackingQuality = m_FrameTracker.tracking_stability();
        } else {
            // Use null motion but continue processing
            context.motion = m_NullMotion;
            context.trackingQuality = 0.0f;
            // Note: This is not necessarily an error, just degraded tracking
        }

        context.shouldStabilize = m_Settings.stabilize_output;
        return true;
    }

//---------------------------------------------------------------------------------------------------------------------

    void StabilizationFilter::updateQualityMetrics(float trackingQuality)
    {
        float newSceneQuality = exp_moving_average(
            m_SceneQuality.load(), trackingQuality, QA_UPDATE_RATE);
        m_SceneQuality.store(newSceneQuality);
        float newTrustFactor = m_TrustFactor.load();
        if (trackingQuality < m_Settings.min_tracking_quality) {
            newTrustFactor = 0.0f;
        } else if (newSceneQuality < m_Settings.min_scene_quality) {
            newTrustFactor = step(newTrustFactor, 0.0f, QA_BLEND_STEP);
        } else {
            newTrustFactor = step(newTrustFactor, 1.0f, QA_BLEND_STEP);
        }
        m_TrustFactor.store(newTrustFactor);
    }

//---------------------------------------------------------------------------------------------------------------------

    MotionVector StabilizationFilter::applyQualityControl(const MotionVector& motion) const
    {
        return motion * m_TrustFactor.load();
    }

//---------------------------------------------------------------------------------------------------------------------

    void StabilizationFilter::processStabilizedOutput(const MotionVector& correctedMotion, VideoFrame& output)
    {
        if (!ready()) {
            output.release();
            return;
        }
        auto& nextFrame = m_FrameQueue.oldest();
        m_FrameQueue.skip();
        auto correction = correctedMotion;
        if (m_Settings.crop_to_stable_region) {
            correction += m_PathSmoother.scene_crop();
        }
        correction.apply(nextFrame, output, m_Settings.background_colour);
    }

//---------------------------------------------------------------------------------------------------------------------

    void StabilizationFilter::handlePassthroughMode(VideoFrame&& input, VideoFrame& output)
    {
        m_FrameQueue.push(std::move(input));
        if (ready()) {
            std::swap(output, m_FrameQueue.oldest());
            m_FrameQueue.skip(1);
            if (m_Settings.crop_to_stable_region) {
                m_PathSmoother.scene_crop().apply(output, m_WarpFrame);
                std::swap(output, m_WarpFrame);
            }
        } else {
            output.release();
        }
    }

//---------------------------------------------------------------------------------------------------------------------

	void StabilizationFilter::filter(VideoFrame&& input, VideoFrame& output)
	{
        std::lock_guard<std::mutex> processingLock(m_processingMutex);
        std::shared_lock<std::shared_mutex> configLock(m_configMutex);

        if (m_IsProcessing.exchange(true)) {
            output.release();
            return;
        }

        auto guard = std::unique_ptr<void, std::function<void(void*)>>(
            (void*)1,
            [this](void*) { m_IsProcessing = false; }
        );

        LVK_ASSERT(input.has_known_format());
        LVK_ASSERT(!input.empty());

        ensureBuffersAllocated(input.size());

        if (!m_Settings.stabilize_output) {
            handlePassthroughMode(std::move(input), output);
            return;
        }

        ProcessingContext context;
        std::string errorMsg;
        if (!prepareFrame(input, context, errorMsg)) {
            lvk::context::log_error("Frame preparation failed: " + errorMsg);
            output.release();
            return;
        }

        updateQualityMetrics(context.trackingQuality);
        auto controlledMotion = applyQualityControl(context.motion);
        m_FrameQueue.push(std::move(input));

        if (ready()) {
            auto correction = m_PathSmoother.next(controlledMotion);
            processStabilizedOutput(correction, output);
        } else {
            output.release();
        }
	}

//---------------------------------------------------------------------------------------------------------------------

	void StabilizationFilter::restart()
	{
        m_SceneQuality.store(1.0f);
        m_FrameQueue.clear();
        reset_context();
	}

//---------------------------------------------------------------------------------------------------------------------

    bool StabilizationFilter::ready() const
    {
        return m_FrameQueue.is_full();
    }

//---------------------------------------------------------------------------------------------------------------------

	void StabilizationFilter::reset_context()
	{
		m_FrameTracker.restart();
        m_PathSmoother.restart();
	}

//---------------------------------------------------------------------------------------------------------------------

    void StabilizationFilter::draw_trackers()
    {
        auto& frame = m_FrameQueue.newest();
        m_FrameTracker.draw_trackers(
            frame,
            lerp<cv::Scalar,double>(
                col::RED[frame.format],
                col::GREEN[frame.format],
                m_TrustFactor.load()
            ),
            7, 10
        );
    }

//---------------------------------------------------------------------------------------------------------------------

    void StabilizationFilter::draw_motion_mesh()
    {
        auto& frame = m_FrameQueue.newest();
        draw_grid(
            frame,
            m_Settings.motion_resolution - cv::Size{1,1},
            col::BLUE[frame.format],
            1
        );
    }

//---------------------------------------------------------------------------------------------------------------------

    size_t StabilizationFilter::frame_delay() const
    {
        return m_PathSmoother.time_delay();
    }

//---------------------------------------------------------------------------------------------------------------------

	cv::Rect StabilizationFilter::stable_region() const
	{
        const auto& margins = m_PathSmoother.scene_margins();
        const auto& frame_size = cv::Size2f(m_FrameQueue.oldest().size());

        return {margins.tl() * frame_size, margins.size() * frame_size};
	}

//---------------------------------------------------------------------------------------------------------------------

}