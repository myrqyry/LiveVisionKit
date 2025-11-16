#pragma once

#include <obs-module.h>
#include "LiveVisionKit.hpp"
#include "Vision/ObjectDetector.hpp"

namespace lvk {
    class ODFilter {
    public:
        static obs_properties_t* Properties();
        static void LoadDefaults(obs_data_t* settings);

        explicit ODFilter(obs_source_t* context);
        ~ODFilter();

        void render();
        void configure(obs_data_t* settings);
        bool validate() const;

    private:
        obs_source_t* m_Context = nullptr;
        std::unique_ptr<vision::ObjectDetector> m_ObjectDetector;
    };
}
