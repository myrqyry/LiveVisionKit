#include "ObjectDetector.hpp"

#include <fstream>
#include <iostream>

namespace lvk {
    namespace vision {
        namespace {
            std::vector<std::string> load_labels(const std::string& path) {
                std::vector<std::string> labels;
                std::ifstream file(path);
                if (file.is_open()) {
                    std::string line;
                    while (std::getline(file, line)) {
                        labels.push_back(line);
                    }
                    file.close();
                }
                return labels;
            }
        }

        ObjectDetector::ObjectDetector(const std::string& model_path, const std::string& label_path) {
            m_model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
            if (!m_model) {
                std::cerr << "Failed to load model" << std::endl;
                return;
            }

            tflite::ops::builtin::BuiltinOpResolver resolver;
            tflite::InterpreterBuilder(*m_model, resolver)(&m_interpreter);
            if (!m_interpreter) {
                std::cerr << "Failed to construct interpreter" << std::endl;
                return;
            }

            if (m_interpreter->AllocateTensors() != kTfLiteOk) {
                std::cerr << "Failed to allocate tensors" << std::endl;
                return;
            }

            m_labels = load_labels(label_path);
        }

        ObjectDetector::~ObjectDetector() {
        }

        void ObjectDetector::detect(cv::Mat& frame) {
            // Get input tensor
            TfLiteTensor* input_tensor = m_interpreter->input_tensor(0);

            // Resize frame
            cv::Mat resized_frame;
            cv::resize(frame, resized_frame, cv::Size(input_tensor->dims->data[2], input_tensor->dims->data[1]));

            // Copy data to input tensor
            memcpy(input_tensor->data.uint8, resized_frame.data, resized_frame.total() * resized_frame.elemSize());

            // Run inference
            if (m_interpreter->Invoke() != kTfLiteOk) {
                std::cerr << "Failed to invoke interpreter" << std::endl;
                return;
            }

            // Get output tensors
            const float* detection_boxes = m_interpreter->tensor(m_interpreter->outputs()[0])->data.f;
            const float* detection_classes = m_interpreter->tensor(m_interpreter->outputs()[1])->data.f;
            const float* detection_scores = m_interpreter->tensor(m_interpreter->outputs()[2])->data.f;
            const float* num_detections = m_interpreter->tensor(m_interpreter->outputs()[3])->data.f;

            // Draw bounding boxes
            for (int i = 0; i < *num_detections; ++i) {
                if (detection_scores[i] > 0.5) {
                    int class_id = static_cast<int>(detection_classes[i]);
                    float ymin = detection_boxes[i * 4] * frame.rows;
                    float xmin = detection_boxes[i * 4 + 1] * frame.cols;
                    float ymax = detection_boxes[i * 4 + 2] * frame.rows;
                    float xmax = detection_boxes[i * 4 + 3] * frame.cols;

                    cv::rectangle(frame, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(0, 255, 0), 2);
                    cv::putText(frame, m_labels[class_id], cv::Point(xmin, ymin - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
                }
            }
        }
    }
}
