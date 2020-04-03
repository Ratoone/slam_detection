//
// Created by ratoone on 03-04-20.
//

#ifndef SLAM_DETECTION_VIDEOSTREAM_H
#define SLAM_DETECTION_VIDEOSTREAM_H


#include <string>
#include <opencv2/videoio.hpp>

class VideoStream {
private:
    cv::VideoCapture video;
public:
    explicit VideoStream(const std::string& videoPath);

    std::optional<cv::Mat> getNextFrame();

    ~VideoStream();
};


#endif //SLAM_DETECTION_VIDEOSTREAM_H
