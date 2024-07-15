#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <chrono>
#include <opencv2/text.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

#include <string>
 tesseract::TessBaseAPI *tess = new tesseract::TessBaseAPI();

std::string recognizeText(const cv::Mat& image) {
    // Convert the image to grayscale
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    // Optional: Apply additional preprocessing if needed
    cv::threshold(gray, gray, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Set the image in Tesseract
    tess->SetImage(gray.data, gray.cols, gray.rows, gray.channels(), gray.step);
    tess->SetSourceResolution(70);

    // Perform OCR
    char* outText = tess->GetUTF8Text();
    std::string result(outText);

    // Clean up
    delete[] outText;

    return result;
}
void detectText(cv::Mat& frame, cv::dnn::Net& net) {
    // Prepare the input blob
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(320, 320), cv::Scalar(123.68, 116.78, 103.94), true, false);
    // Set the input blob
    net.setInput(blob);
    // Forward pass to get output
    std::vector<cv::Mat> output;
    std::vector<cv::String> outputNames = { "feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3" };
    net.forward(output, outputNames);

    // Extract results from output
    cv::Mat scores = output[0];
    cv::Mat geometry = output[1];

    // Decode the detections
    std::vector<cv::RotatedRect> boxes;
    std::vector<float> confidences;
    float scoreThreshold = 0.9;
    float nmsThreshold = 0.4;

    for (int y = 0; y < scores.size[2]; y++) {
        const float* scoresData = scores.ptr<float>(0, 0, y);
        const float* x0Data = geometry.ptr<float>(0, 0, y);
        const float* x1Data = geometry.ptr<float>(0, 1, y);
        const float* x2Data = geometry.ptr<float>(0, 2, y);
        const float* x3Data = geometry.ptr<float>(0, 3, y);
        const float* anglesData = geometry.ptr<float>(0, 4, y);

        for (int x = 0; x < scores.size[3]; x++) {
            float score = scoresData[x];
            if (score >= scoreThreshold) {
                float offsetX = x * 4.0f;
                float offsetY = y * 4.0f;
                float angle = anglesData[x];
                float cosA = cos(angle);
                float sinA = sin(angle);
                float h = x0Data[x] + x2Data[x];
                float w = x1Data[x] + x3Data[x];
                cv::Point2f offset(offsetX + cosA * x1Data[x] + sinA * x2Data[x],
                                   offsetY - sinA * x1Data[x] + cosA * x2Data[x]);
                cv::Point2f p1 = cv::Point2f(-sinA * h + offset.x, -cosA * h + offset.y);
                cv::Point2f p3 = cv::Point2f(-cosA * w + offset.x, sinA * w + offset.y);
                cv::RotatedRect rect = cv::RotatedRect(0.5f * (p1 + p3), cv::Size2f(w, h), -angle * 180.0f / CV_PI);
                boxes.push_back(rect);
                confidences.push_back(score);
            }
        }
    }

    // Apply non-maximum suppression
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, scoreThreshold, nmsThreshold, indices);
    // Scale back the bounding boxes to match the original frame size
    float scale_x = static_cast<float>(frame.cols)/ 320.0f;
    float scale_y = static_cast<float>(frame.rows)/ 320.0f;

    // Draw the boxes on the original frame
    for (size_t i = 0; i < indices.size() ; ++i) {
        
        cv::RotatedRect& box = boxes[indices[i]];
        box.center.x *= scale_x;
        box.center.y *= scale_y;
        box.size.width *= scale_x;
        box.size.height *= scale_y;

        cv::Point2f vertices[4];
        box.points(vertices);
        for (int j = 0; j < 4; ++j) {
            cv::line(frame, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }
        cv::Rect roi = box.boundingRect();
        roi &= cv::Rect(0, 0, frame.cols, frame.rows);  // Ensure ROI is within frame bounds
        cv::Mat cropped = frame(roi);

        // Recognize text
        if (tess->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY)) {
        std::cerr << "Could not initialize tesseract." << std::endl;
        return;
        }

        std::string text = recognizeText(cropped);

        // Draw recognized text
        cv::putText(frame, text, box.center, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    }
}

int main() {
    // Load the pre-trained EAST model
    cv::dnn::Net net = cv::dnn::readNet("/home/ahaanbanerjee/SLAM_OCR/frozen_east_text_detection.pb");

    // Open the default camera (0). Change the index if necessary.
    cv::VideoCapture cap(2);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        return -1;
    }

    cv::Mat frame;
    int frame_count = 0;
    double fps = 0.0;
    auto start_time = std::chrono::steady_clock::now();

    while (true) {
        auto start_loop_time = std::chrono::steady_clock::now();

        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Failed to capture frame" << std::endl;
            break;
        }

        // Detect text and draw bounding boxes
        detectText(frame, net);

        // Calculate FPS
        frame_count++;
        auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start_time).count();
        if (elapsed_time >= 1) {
            fps = frame_count / static_cast<double>(elapsed_time);
            frame_count = 0;
            start_time = std::chrono::steady_clock::now();
        }

        // Put FPS text on the frame
        std::string fps_text = "FPS: " + std::to_string(fps);
        cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);

        // Display the frame
        cv::imshow("Camera Feed", frame);

        // Frame time for debugging
     
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
