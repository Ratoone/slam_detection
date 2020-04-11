//
// Created by ratoone on 03-04-20.
//

#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <utility>
#include "Matching.h"
#define CERES_FOUND 1
#include <opencv2/sfm/reconstruct.hpp>

Matching::Matching(cv::Mat cameraMatrix) : cameraMatrix{std::move(cameraMatrix)}{}

std::vector<cv::Point2f> Matching::featureDetection(const cv::Mat& image) {
    std::vector<cv::KeyPoint> keyPoints;
    auto detector = cv::FastFeatureDetector::create();
    detector->detect(image, keyPoints);
    std::vector<cv::Point2f> points;
    cv::KeyPoint::convert(keyPoints, points, std::vector<int>());
    return points;
}

cv::Mat Matching::featureMatching(const cv::Mat& source, const cv::Mat& target){
    std::vector<float> error;
    auto sourcePoints = featureDetection(source);
    auto targetPoints = featureDetection(target);
    std::vector<uchar> status;
    //remove outliers
    cv::calcOpticalFlowPyrLK(source, target, sourcePoints, targetPoints, status, error);
    int indexCorrection = 0;
    for( int i=0; i < status.size(); i++){
        if (status.at(i) == 0)	{
            sourcePoints.erase (sourcePoints.begin() + (i - indexCorrection));
            targetPoints.erase (targetPoints.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }

    auto transformation =  findTransformation(sourcePoints, targetPoints);
    cv::Mat projectInitial = cameraMatrix * cv::Mat(cv::Matx34d(
            rotation.at<double>(0,0), rotation.at<double>(0,1),rotation.at<double>(0,2),translation.at<double>(0),
            rotation.at<double>(1,0), rotation.at<double>(1,1),rotation.at<double>(1,2),translation.at<double>(1),
            rotation.at<double>(2,0), rotation.at<double>(2,1),rotation.at<double>(2,2),translation.at<double>(2)
            ));

    translation = translation + rotation * transformation.second;
    rotation = transformation.first * rotation;

    cv::Mat projectFinal = cameraMatrix * cv::Mat(cv::Matx34d(
            rotation.at<double>(0,0), rotation.at<double>(0,1),rotation.at<double>(0,2),translation.at<double>(0),
            rotation.at<double>(1,0), rotation.at<double>(1,1),rotation.at<double>(1,2),translation.at<double>(1),
            rotation.at<double>(2,0), rotation.at<double>(2,1),rotation.at<double>(2,2),translation.at<double>(2)
    ));

    if (shouldCompute3D) {
        cv::Mat points4D, points3D;
        cv::triangulatePoints(projectInitial, projectFinal, sourcePoints, targetPoints, points4D);
        cv::Mat points4DChanneled = points4D.reshape(4, 1);
        cv::convertPointsFromHomogeneous(points4DChanneled, points3D);
        return points3D;
    }
    return {};
}

std::pair<cv::Mat, cv::Mat> Matching::findTransformation(std::vector<cv::Point2f>& sourcePoints, std::vector<cv::Point2f>& targetPoints){
    auto essentialMatrix = cv::findEssentialMat(sourcePoints, targetPoints, cameraMatrix);
    cv::Mat relRotation, relTranslation;
    cv::recoverPose(essentialMatrix, sourcePoints, targetPoints, cameraMatrix, relRotation, relTranslation);
    return {relRotation, relTranslation};
}

const cv::Mat &Matching::getRotation() const {
    return rotation;
}

const cv::Mat &Matching::getTranslation() const {
    return translation;
}
