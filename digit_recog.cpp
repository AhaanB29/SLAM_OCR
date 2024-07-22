#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

// Function to preprocess the image
cv::Mat preprocess(const cv::Mat& image) {
    cv::Mat gray, blurred, binary;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    cv::adaptiveThreshold(blurred, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2);
    return binary;
}
void digitRecog(cv::Mat& image, cv::dnn::Net& net)
{
    cv::Mat binary = preprocess(image);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> validBoxes;
    std::vector<int> digits;
    std::vector<float> confidences;

    for (const auto& contour : contours) {
        cv::Rect boundingBox = cv::boundingRect(contour);
        
        // Filter based on aspect ratio and size
        float aspect_ratio = (float)boundingBox.width / boundingBox.height;
        if (aspect_ratio < 0.2 || aspect_ratio > 1.5 || 
            boundingBox.width < 10 || boundingBox.height < 20 || 
            boundingBox.width > image.cols / 3 || boundingBox.height > image.rows / 3) {
            continue;
        }

        cv::Mat roi = binary(boundingBox);
        cv::Mat resizedRoi;
        cv::resize(roi, resizedRoi, cv::Size(28, 28));

        cv::Mat blob = cv::dnn::blobFromImage(resizedRoi, 1.0/255.0, cv::Size(28, 28), cv::Scalar(0), true, false);

        net.setInput(blob);
        cv::Mat output = net.forward();

        cv::Point classIdPoint;
        double confidence;
        cv::minMaxLoc(output, nullptr, &confidence, nullptr, &classIdPoint);
        int digit = classIdPoint.x;

        // Only consider if confidence is high enough
        if (confidence > 0.7) {
            validBoxes.push_back(boundingBox);
            digits.push_back(digit);
            confidences.push_back(static_cast<float>(confidence));
        }
    }

    // Perform NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(validBoxes, confidences, 0.5, 0.4, indices);

    // Draw only the boxes that survived NMS
    for (int idx : indices) {
        cv::rectangle(image, validBoxes[idx], cv::Scalar(0, 255, 0), 2);
        cv::putText(image, std::to_string(digits[idx]), validBoxes[idx].tl(), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
    }
}
int main() {
    // Load the image
    cv::dnn::Net net = cv::dnn::readNetFromONNX("/home/pc/ORB_SLAM2/SLAM_OCR/digit_recognition_model.onnx");
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
        digitRecog(frame, net);

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

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}