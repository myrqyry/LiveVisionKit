#include "ODFilter.hpp"
#include "Utility/ScopedProfiler.hpp"
#include "Utility/Locale.hpp"
#include "Interop/FrameIngest.hpp"

namespace lvk {
    obs_properties_t* ODFilter::Properties() {
        obs_properties_t* properties = obs_properties_create();
        return properties;
    }

    void ODFilter::LoadDefaults(obs_data_t* settings) {
        // No defaults to load
    }

    ODFilter::ODFilter(obs_source_t* context) : m_Context(context) {
        m_ObjectDetector = std::make_unique<vision::ObjectDetector>("Models/detect.tflite", "Models/labelmap.txt");
    }

    ODFilter::~ODFilter() {
    }

    void ODFilter::render() {
        LVK_PROFILE;
        obs_source_t* parent = obs_filter_get_parent(m_Context);
        if (!parent) {
            obs_source_skip_video_filter(m_Context);
            return;
        }

        obs_source_frame* frame = obs_source_get_frame(parent);
        if (!frame) {
            obs_source_skip_video_filter(m_Context);
            return;
        }

        auto ingest = FrameIngest::Select(frame->format);
        if (!ingest) {
            obs_source_skip_video_filter(m_Context);
            return;
        }

        VideoFrame video_frame;
        ingest->upload_obs_frame(frame, video_frame);

        cv::Mat mat = video_frame.getMat(cv::ACCESS_READ);
        m_ObjectDetector->detect(mat);

        ingest->download_ocl_frame(video_frame, frame);
        obs_source_release_frame(parent, frame);
    }

    void ODFilter::configure(obs_data_t* settings) {
        // No settings to configure
    }

    bool ODFilter::validate() const {
        return m_Context != nullptr;
    }
}
