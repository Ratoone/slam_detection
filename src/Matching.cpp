//
// Created by ratoone on 03-04-20.
//

#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <utility>
#include "Matching.h"

Matching::Matching(cv::Mat cameraMatrix) : cameraMatrix{std::move(cameraMatrix)}{}

std::vector<cv::Point2f> Matching::featureDetection(const cv::Mat& image) {
    std::vector<cv::KeyPoint> keyPoints;
    auto detector = cv::FastFeatureDetector::create();
    detector->detect(image, keyPoints);
    std::vector<cv::Point2f> points;
    cv::KeyPoint::convert(keyPoints, points, std::vector<int>());
    return points;
}

std::pair<cv::Mat, cv::Mat> Matching::featureMatching(const cv::Mat& source, const cv::Mat& target){
    std::vector<float> error;
    auto sourcePoints = featureDetection(source);
    auto targetPoints = featureDetection(target);
    std::vector<uchar> status;
    cv::calcOpticalFlowPyrLK(source, target, sourcePoints, targetPoints, status, error);

    int indexCorrection = 0;
    for( int i=0; i<status.size(); i++){
        cv::Point2f pt = targetPoints.at(i- indexCorrection);
        if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))	{
            if((pt.x<0)||(pt.y<0))	{
                status.at(i) = 0;
            }
            sourcePoints.erase (sourcePoints.begin() + i - indexCorrection);
            targetPoints.erase (targetPoints.begin() + i - indexCorrection);
            indexCorrection++;
        }

    }

    return findTransformation(sourcePoints, targetPoints);
}

std::pair<cv::Mat, cv::Mat> Matching::findTransformation(std::vector<cv::Point2f>& sourcePoints, std::vector<cv::Point2f>& targetPoints){
    auto essentialMatrix = cv::findEssentialMat(sourcePoints, targetPoints, cameraMatrix);
    cv::Mat rotation, translation;
    cv::recoverPose(essentialMatrix, sourcePoints, targetPoints, cameraMatrix, rotation, translation);
    return {rotation, translation};
}
