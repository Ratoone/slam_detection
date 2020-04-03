//
// Created by ratoone on 03-04-20.
//

#ifndef SLAM_DETECTION_MATCHING_H
#define SLAM_DETECTION_MATCHING_H


#include <opencv2/core/mat.hpp>

class Matching {
private:
    cv::Mat cameraMatrix;
public:
    explicit Matching(cv::Mat cameraMatrix);

    std::vector<cv::Point2f> featureDetection(const cv::Mat& image);

    std::pair<cv::Mat, cv::Mat> featureMatching(const cv::Mat &source, const cv::Mat &target);

    std::pair<cv::Mat, cv::Mat> findTransformation(std::vector<cv::Point2f> &sourcePoints, std::vector<cv::Point2f> &targetPoints);
};


#endif //SLAM_DETECTION_MATCHING_H
