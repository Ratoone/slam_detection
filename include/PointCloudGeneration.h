//
// Created by ratoone on 08-04-20.
//

#ifndef SLAM_DETECTION_POINTCLOUDGENERATION_H
#define SLAM_DETECTION_POINTCLOUDGENERATION_H


#include <opencv2/core/mat.hpp>

class PointCloudGeneration {
private:
    cv::Mat cameraMatrix;
    cv::Mat distortion;
    cv::Ptr<cv::StereoSGBM> stereoMatcher;
    cv::Mat Q;
public:
    explicit PointCloudGeneration(const cv::Mat &cameraMatrix);

    cv::Mat generatePointCloud(const cv::Mat &sourceImage, const cv::Mat &targetImage, const cv::Mat &rotation, const cv::Mat &translation);

    std::pair<cv::Mat, cv::Mat> rectifyImage(const cv::Mat &sourceImage, const cv::Mat &targetImage, const cv::Mat &rotation,
                                             const cv::Mat &translation);
};


#endif //SLAM_DETECTION_POINTCLOUDGENERATION_H
