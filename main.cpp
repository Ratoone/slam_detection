//
// Created by ratoone on 02-04-20.
//
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <Calibration.h>
#include <VideoStream.h>
#include <Matching.h>
#include <ObjectDetection.h>
#include <filesystem>

int main(){
    Calibration calib = Calibration("../parameter.dat", "../images/calibration");
    VideoStream video = VideoStream("../images/MVI_3820.MP4");
    Matching matching(calib.getCameraMatrix().clone());
    ObjectDetection objectDetection;
    auto videoWriter = cv::VideoWriter("../images/output.MP4",cv::VideoWriter::fourcc('M','P','4','V'),25,cv::Size(1400,600));

    cv::Mat trajectory = cv::Mat::zeros(600, 600, CV_8UC3);
    cv::arrowedLine(trajectory,cv::Point(10, 500), cv::Point(10, 550), CV_RGB(255, 255, 255));
    cv::putText(trajectory, "Z", cv::Point(20, 550), cv::FONT_HERSHEY_PLAIN,1,CV_RGB(255,255,255));
    cv::arrowedLine(trajectory,cv::Point(10, 500), cv::Point(60, 500), CV_RGB(255, 255, 255));
    cv::putText(trajectory, "X", cv::Point(70, 500), cv::FONT_HERSHEY_PLAIN,1,CV_RGB(255,255,255));

    cv::Mat frame = video.getNextFrame().value();
    std::optional<cv::Mat> next = video.getNextFrame();
    std::vector<cv::Vec3f> displayPointCloud;
    while(next){
        frame = calib.rectifyImage(frame);
        cv::Mat nextFrame = calib.rectifyImage(next.value());

        matching.featureMatching(frame, nextFrame);
        auto rotation = matching.getRotation();
        auto translation = matching.getTranslation();

        objectDetection.detectImage(frame);

        int x = int(translation.at<double>(0)) + 300;
        int y = int(translation.at<double>(2)) + 300;
        cv::circle(trajectory, cv::Point(x, y) ,1, CV_RGB(255,0,0), 2);
        char *coordinates = new char[255];
        sprintf(coordinates, "X: %.2f, Y: %.2f, Z: %.2f",translation.at<double>(0),translation.at<double>(1), translation.at<double>(2));
        cv::rectangle(trajectory, cv::Point(0,0), cv::Point(trajectory.cols,100),CV_RGB(0,0,255), -1);
        cv::putText(trajectory, coordinates, cv::Point(10, 50), cv::FONT_HERSHEY_PLAIN,1,CV_RGB(255,255,255));

        // merge the two images
        cv::Mat combinedFrame(std::max(frame.rows, trajectory.rows), frame.cols + trajectory.cols, CV_8UC3);
        cv::Mat leftFragment(combinedFrame, cv::Rect(0, 0, frame.size().width, frame.size().height));
        frame.copyTo(leftFragment);
        cv::Mat rightFragment(combinedFrame, cv::Rect(frame.size().width, 0, trajectory.size().width, trajectory.size().height));
        trajectory.copyTo(rightFragment);

        cv::imshow("", combinedFrame);
        cv::waitKey(1);
        videoWriter.write(combinedFrame);
        frame = next.value();
        next = video.getNextFrame();
    }
}