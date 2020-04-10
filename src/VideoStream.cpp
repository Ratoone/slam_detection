//
// Created by ratoone on 03-04-20.
//

#include "VideoStream.h"
#include <filesystem>
#include <opencv2/imgcodecs.hpp>

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

void VideoStream::resetVideo() {
    video.set(cv::CAP_PROP_POS_FRAMES, 0);
}

int VideoStream::getFrameCount(){
    return int(video.get(cv::CAP_PROP_FRAME_COUNT));
}

VideoStream::~VideoStream(){
    video.release();
}