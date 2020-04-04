//
// Created by ratoone on 02-04-20.
//

#include <opencv2/opencv.hpp>
#include <Calibration.h>
#include <VideoStream.h>
#include <Matching.h>
#include <ObjectDetection.h>

int main(){
    Calibration calib = Calibration("../parameter.dat", "../images/calibration");
    VideoStream video = VideoStream("../images/MVI_3777.MP4");
    Matching matching(calib.getCameraMatrix().clone());
    ObjectDetection objectDetection;
    cv::Mat frame = video.getNextFrame().value();
    std::optional<cv::Mat> next = video.getNextFrame();

    cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat translation = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat trajectory = cv::Mat::zeros(600, 600, CV_8UC3);

    while(next){
        frame = calib.rectifyImage(frame);
        cv::Mat nextFrame = calib.rectifyImage(next.value());
//        auto transformation = matching.featureMatching(frame, nextFrame);
//        translation = translation + rotation * transformation.second;
//        rotation = transformation.first * rotation;
//        int x = int(translation.at<double>(0)) + 300;
//        int y = int(translation.at<double>(2)) + 100;
//        cv::circle(trajectory, cv::Point(x, y) ,1, CV_RGB(255,0,0), 2);
//
//        cv::imshow("trajectory", trajectory);
        objectDetection.detectImage(frame);
        cv::imshow("", frame);
        cv::waitKey(10);
        frame = next.value();
        next = video.getNextFrame();
    }
}