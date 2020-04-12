//
// Created by ratoone on 04-04-20.
//

#include <fstream>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "ObjectDetection.h"

ObjectDetection::ObjectDetection() {
    std::string classesFile = "../coco.names";
    std::ifstream ifs(classesFile.c_str());
    std::string line;
    while (getline(ifs, line)) {
        classes.push_back(line);
    }

    std::string modelConfig = "../yolov3.cfg";
    std::string modelWeights = "../yolov3.weights";
    net = cv::dnn::readNetFromDarknet(modelConfig, modelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    std::vector<int> outLayers = net.getUnconnectedOutLayers();
    std::vector<std::string> layersNames = net.getLayerNames();

    // Get the names of the output layers in names
    labelNames.resize(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i) {
        labelNames[i] = layersNames[outLayers[i] - 1];
    }
}

void ObjectDetection::detectImage(cv::Mat &image){
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0/255, cv::Size(inpWidth, inpHeight));
    net.setInput(blob);
    std::vector<cv::Mat> outputLabels;
    net.forward(outputLabels, labelNames);
    boundingBoxPostprocess(image, outputLabels);
}

void ObjectDetection::boundingBoxPostprocess(cv::Mat& frame, const std::vector<cv::Mat>& boundingBoxes)
{
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // consider only the bounding boxes with confidence bigger than the threshold
    for (const auto & boundingBox : boundingBoxes){
        auto* data = (float*)boundingBox.data;
        for (int j = 0; j < boundingBox.rows; ++j, data += boundingBox.cols){
            cv::Mat scores = boundingBox.row(j).colRange(5, boundingBox.cols);
            cv::Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            cv::minMaxLoc(scores, nullptr, &confidence, nullptr, &classIdPoint);
            if (confidence < confThreshold){
                continue;
            }
            int centerX = (int)(data[0] * frame.cols);
            int centerY = (int)(data[1] * frame.rows);
            int width = (int)(data[2] * frame.cols);
            int height = (int)(data[3] * frame.rows);
            int left = centerX - width / 2;
            int top = centerY - height / 2;

            classIds.push_back(classIdPoint.x);
            confidences.push_back((float)confidence);
            boxes.emplace_back(left, top, width, height);
        }
    }

    // perform non maximum suppression to eliminate redundant overlapping boxes
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (int idx : indices)
    {
        cv::Rect box = boxes[idx];
        if (classes[classIds[idx]] != "bicycle"){
            continue;
        }
        drawBoundingBox(classIds[idx], confidences[idx], box.x, box.y,
                        box.x + box.width, box.y + box.height, frame);
    }
}

void ObjectDetection::drawBoundingBox(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)
{
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 178, 50), 3);

    std::string label = cv::format("%.2f", conf);
    if (!classes.empty()){
        assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = std::max(top, labelSize.height);
    cv::rectangle(frame, cv::Point(left, top - round(1.5*labelSize.height)), cv::Point(left + round(1.5*labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0),1);
}