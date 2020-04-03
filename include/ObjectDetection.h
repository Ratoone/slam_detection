//
// Created by ratoone on 04-04-20.
//

#ifndef SLAM_DETECTION_OBJECTDETECTION_H
#define SLAM_DETECTION_OBJECTDETECTION_H


#include <opencv2/dnn.hpp>

class ObjectDetection {
private:
    // Initialize the parameters
    float confThreshold = 0.5; // Confidence threshold
    float nmsThreshold = 0.4;  // Non-maximum suppression threshold
    int inpWidth = 1920;        // Width of network's input image
    int inpHeight = 1080;       // Height of network's input image
    cv::dnn::Net net;
    std::vector<std::string> classes;
public:
    ObjectDetection();

    void detectImage(const cv::Mat &image);
};


#endif //SLAM_DETECTION_OBJECTDETECTION_H
