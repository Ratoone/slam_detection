//
// Created by ratoone on 03-04-20.
//

#include "VideoStream.h"
#include <filesystem>

VideoStream::VideoStream(const std::string &videoPath) {
    assert(std::filesystem::exists(videoPath));
    video.open(videoPath);
    assert(video.isOpened());
}

std::optional<cv::Mat> VideoStream::getNextFrame(){
    cv::Mat frame;
    video >> frame;
    if (frame.empty()){
        return std::nullopt;
    }

    return frame;
}

VideoStream::~VideoStream(){
    video.release();
}