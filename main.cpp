//
// Created by ratoone on 02-04-20.
//

#include <opencv2/opencv.hpp>
#include <Calibration.h>
#include <VideoStream.h>

int main(){
    Calibration calib = Calibration("../parameter.dat", "../images/calibration");
    VideoStream video = VideoStream("../images/MVI_3777.MP4");

    cv::Mat frame = video.getNextFrame().value();
    std::optional<cv::Mat> nextFrame = video.getNextFrame();
    while(nextFrame){
        frame = calib.rectifyImage(frame);
        cv::imshow("", frame);
        cv::waitKey();
        frame = nextFrame.value();
        nextFrame = video.getNextFrame();
    }
}