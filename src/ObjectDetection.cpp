//
// Created by ratoone on 04-04-20.
//

#include <fstream>
#include "ObjectDetection.h"

ObjectDetection::ObjectDetection() {
    std::string classesFile = "coco.names";
    std::ifstream ifs(classesFile.c_str());
    std::string line;
    while (getline(ifs, line)) classes.push_back(line);

    std::string modelConfig = "../yolov3.cfg";
    std::string modelWeights = "../yolov3.weights";
    net = cv::dnn::readNetFromDarknet(modelConfig, modelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

void ObjectDetection::detectImage(const cv::Mat &image){
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob);
    net.setInput(blob);
    std::vector<cv::Mat> output;
    net.forward();
}
