#pragma once

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "opencv2/opencv.hpp"

#include <string>
#include <vector>

namespace lvk {
    namespace vision {
        class ObjectDetector {
        public:
            ObjectDetector(const std::string& model_path, const std::string& label_path);
            ~ObjectDetector();

            void detect(cv::Mat& frame);

        private:
            std::unique_ptr<tflite::FlatBufferModel> m_model;
            std::unique_ptr<tflite::Interpreter> m_interpreter;
            std::vector<std::string> m_labels;
        };
    }
}
