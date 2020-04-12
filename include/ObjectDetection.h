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
    int inpWidth = 416;        // Width of network's input image
    int inpHeight = 416;       // Height of network's input image
    cv::dnn::Net net;
    std::vector<std::string> classes;
    std::vector<std::string> labelNames;
public:
    ObjectDetection();

    void boundingBoxPostprocess(cv::Mat &frame, const std::vector<cv::Mat> &boundingBoxes);

    void drawBoundingBox(int classId, float conf, int left, int top, int right, int bottom, cv::Mat &frame);

    void detectImage(cv::Mat &image);
};


#endif //SLAM_DETECTION_OBJECTDETECTION_H
