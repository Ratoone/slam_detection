//
// Created by ratoone on 02-04-20.
//

#include <opencv2/opencv.hpp>
#include "Calibration.h"
#include <filesystem>

Calibration::Calibration(const std::string& parameterFilePath, const std::string& calibrationImagePath) {
    if (std::filesystem::exists(parameterFilePath)){
        loadCalibFromFile(parameterFilePath);
        return;
    }

    assert(!calibrationImagePath.empty());
    calibrate(calibrationImagePath);
    saveCalibToFile(parameterFilePath);
}

void Calibration::calibrate(const std::string& calibrationImagePath) {
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;

    std::vector<cv::Point3f> points;

    for(int i = 0; i < patternSize.height; i++){
        for(int j = 0; j < patternSize.width; j++) {
            points.emplace_back(j, i, 0);
        }
    }

    std::vector<cv::String> images;
    cv::glob(calibrationImagePath, images);
    std::vector<cv::Point2f> corners;
    cv::Mat imageRaw, image;
    int count = 0;

    for (auto &imagePath : images){
        imageRaw = cv::imread(imagePath);
        cv::resize(imageRaw, image, cv::Size(imageWidth, imageHeight),0,0,cv::INTER_AREA);

        bool found = cv::findChessboardCorners(image, patternSize, corners);
        if (!found) {
            continue;
        }
        count++;
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        cv::cornerSubPix(gray, corners, cv::Size(5, 5), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.0001));
        objectPoints.emplace_back(points);
        imagePoints.emplace_back(corners);
    }
    printf("Calibration successful with %d/%zu images", count, images.size());
    cv::Mat R, T;
    cv::calibrateCamera(objectPoints, imagePoints, cv::Size(image.rows, image.cols), cameraMatrix, distortion, R, T);
    newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distortion, image.size(), 0);
}

cv::Mat Calibration::rectifyImage(const cv::Mat &image) {
    cv::Mat resized, undistorted;
    cv::resize(image, resized, cv::Size(imageWidth, imageHeight),0,0,cv::INTER_AREA);
    cv::undistort(resized, undistorted, cameraMatrix, distortion, newCameraMatrix);
    return undistorted;
}

void Calibration::saveCalibToFile(const std::string& filePath){
    cv::FileStorage file(filePath, cv::FileStorage::WRITE);
    file << "cameraMatrix" << cameraMatrix;
    file << "newCameraMatrix" << newCameraMatrix;
    file << "distortion" << distortion;
    file.release();
}

void Calibration::loadCalibFromFile(const std::string& filePath){
    cv::FileStorage file(filePath, cv::FileStorage::READ);
    file["cameraMatrix"] >> cameraMatrix;
    file["newCameraMatrix"] >> newCameraMatrix;
    file["distortion"] >> distortion;
    file.release();
}

const cv::Mat &Calibration::getCameraMatrix() const {
    return cameraMatrix;
}
