#include <iostream>
#include <vector>
#include <getopt.h>

#include <opencv2/opencv.hpp>

#include "inference.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    bool runOnGPU = false;

    // Initialize Inference with the ONNX model
    Inference inf("/home/pc/Digit_recog/best.onnx", cv::Size(640, 640), "/home/pc/Digit_recog/classes10.txt", runOnGPU);

    // Open the default camera (usually the webcam)
    VideoCapture cap(2);
    if (!cap.isOpened())
    {
        cerr << "Error: Could not open camera" << endl;
        return -1;
    }

    // Set the desired frame size
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 640);

    while (true)
    {
        Mat frame;
        cap >> frame; // Capture a new frame

        if (frame.empty())
        {
            cerr << "Error: Could not capture frame" << endl;
            break;
        }

        // Inference starts here...
        cv::resize(frame, frame, cv::Size(640, 640));
        std::vector<Detection> output = inf.runInference(frame);

        int detections = output.size();
        std::cout << "Number of detections: " << detections << std::endl;

        for (int i = 0; i < detections; ++i)
        {
            Detection detection = output[i];

            cv::Rect box = detection.box;
            cv::Scalar color = detection.color;

            // Detection box
            cv::rectangle(frame, box, color, 2);

            // Detection box text
            std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

            cv::rectangle(frame, textBox, color, cv::FILLED);
            cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
        }
        // Inference ends here...

        // Display the frame with detections
        cv::imshow("Inference", frame);

        // Press 'q' to exit the loop
        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
