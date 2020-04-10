//
// Created by ratoone on 03-04-20.
//

#ifndef SLAM_DETECTION_MATCHING_H
#define SLAM_DETECTION_MATCHING_H


#include <opencv2/core/mat.hpp>

class Matching {
private:
    cv::Mat cameraMatrix;
    cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat translation = cv::Mat::zeros(3, 1, CV_64F);
    bool shouldCompute3D = false;
public:
    explicit Matching(cv::Mat cameraMatrix);

    std::vector<cv::Point2f> featureDetection(const cv::Mat& image);

    cv::Mat featureMatching(const cv::Mat &source, const cv::Mat &target);

    std::pair<cv::Mat, cv::Mat> findTransformation(std::vector<cv::Point2f> &sourcePoints, std::vector<cv::Point2f> &targetPoints);

    [[nodiscard]] const cv::Mat &getRotation() const;

    [[nodiscard]] const cv::Mat &getTranslation() const;
};


#endif //SLAM_DETECTION_MATCHING_H
