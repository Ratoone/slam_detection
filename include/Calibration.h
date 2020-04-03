//
// Created by ratoone on 02-04-20.
//

#ifndef SLAM_DETECTION_CALIBRATION_H
#define SLAM_DETECTION_CALIBRATION_H


#include <opencv2/core/mat.hpp>

class Calibration {
private:
    cv::Size patternSize = cv::Size(4, 4);
    cv::Mat cameraMatrix;
    cv::Mat distortion;
    cv::Mat newCameraMatrix;
public:
    explicit Calibration(const std::string &parameterFilePath, const std::string &calibrationImagePath = "");

    void calibrate(const std::string& calibrationImagePath);

    cv::Mat rectifyImage(const cv::Mat& image);

    void saveCalibToFile(const std::string &filePath);

    void loadCalibFromFile(const std::string &filePath);

    const cv::Mat &getCameraMatrix() const;
};


#endif //SLAM_DETECTION_CALIBRATION_H
