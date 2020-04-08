//
// Created by ratoone on 08-04-20.
//

#include <opencv2/opencv.hpp>
#include "PointCloudGeneration.h"

PointCloudGeneration::PointCloudGeneration(const cv::Mat &cameraMatrix) {
    this->cameraMatrix = cameraMatrix;
    this->distortion = cv::Mat::zeros(4,1,CV_8U);
    stereoMatcher = cv::StereoSGBM::create(0,32);
    stereoMatcher->setUniquenessRatio(10);
    stereoMatcher->setSpeckleWindowSize(100);
    stereoMatcher->setSpeckleRange(32);
    stereoMatcher->setDisp12MaxDiff(1);
    stereoMatcher->setMode(cv::StereoSGBM::MODE_HH);
    this->Q = cv::Mat::eye(4,4, CV_32F);
}

std::pair<cv::Mat, cv::Mat> PointCloudGeneration::rectifyImage(const cv::Mat &sourceImage, const cv::Mat &targetImage, const cv::Mat &rotation, const cv::Mat &translation) {
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(cameraMatrix, distortion, cameraMatrix, distortion, sourceImage.size(), rotation, translation, R1, R2, P1, P2, Q);

    cv::Mat rectifyMaps[2][2];
    cv::initUndistortRectifyMap(cameraMatrix, distortion, R1, P1, sourceImage.size(), CV_32FC1, rectifyMaps[0][0], rectifyMaps[0][1]);
    cv::initUndistortRectifyMap(cameraMatrix, distortion, R2, P2, targetImage.size(), CV_32FC1, rectifyMaps[1][0], rectifyMaps[1][1]);

    cv::Mat normalizedSource, normalizedTarget;
    cv::normalize(sourceImage, normalizedSource, 0, 1, cv::NORM_MINMAX, CV_32FC1);
    cv::normalize(targetImage, normalizedTarget, 0, 1, cv::NORM_MINMAX, CV_32FC1);

    cv::Mat outputSource, outputTarget;
    cv::remap(normalizedSource, outputSource, rectifyMaps[0][0], rectifyMaps[0][1], cv::INTER_LINEAR);
    cv::remap(normalizedTarget, outputTarget, rectifyMaps[1][0], rectifyMaps[1][1], cv::INTER_LINEAR);

    return {outputSource, outputTarget};
}

cv::Mat PointCloudGeneration::generatePointCloud(const cv::Mat &sourceImage, const cv::Mat &targetImage,
                                                 const cv::Mat &rotation, const cv::Mat &translation) {
    cv::Mat graySource, grayTarget;
    cv::cvtColor(sourceImage, graySource,cv::COLOR_BGR2GRAY);
    cv::cvtColor(targetImage, grayTarget,cv::COLOR_BGR2GRAY);
    auto rectified = rectifyImage(graySource, grayTarget, rotation, translation);
    cv::Mat disparity, finalDisparity, floatDisparity;
    stereoMatcher->compute(sourceImage, targetImage, disparity);
    disparity.convertTo(finalDisparity, CV_8U);
    finalDisparity.convertTo(floatDisparity, CV_32F, 1.0f/16);
    cv::Mat pointCloud;
    cv::reprojectImageTo3D(finalDisparity, pointCloud, Q, false, CV_32F);
    return pointCloud;
}
