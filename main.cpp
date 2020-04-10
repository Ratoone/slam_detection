//
// Created by ratoone on 02-04-20.
//
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <Calibration.h>
#include <VideoStream.h>
#include <Matching.h>
#include <ObjectDetection.h>
#include <PointCloudGeneration.h>
#include <filesystem>


int main(){
    Calibration calib = Calibration("../parameter.dat", "../images/calibration");
    VideoStream video = VideoStream("../images/MVI_3820.MP4");
    Matching matching(calib.getCameraMatrix().clone());
    PointCloudGeneration pointCloudGeneration(calib.getCameraMatrix().clone());
    ObjectDetection objectDetection;


    cv::Mat trajectory = cv::Mat::zeros(600, 600, CV_8UC3);

//    if (std::filesystem::is_empty(framesFolder)){
//        auto frame = video.getNextFrame();
//        int count = 0;
//        while (frame){
//            auto rectifiedFrame = calib.rectifyImage(frame.value());
//            cv::imwrite(framesFolder+"/"+std::to_string(count)+".jpg", rectifiedFrame);
//            frame = video.getNextFrame();
//            count++;
//        }
//        video.resetVideo();
//    }
//
//    std::vector<std::string> frames;
//    for (auto& framePath : std::filesystem::directory_iterator(framesFolder)){
//        frames.emplace_back(framePath.path().string());
//    }
//
//    std::vector<cv::Mat> rotations, translations, points3D;
//    cv::sfm::reconstruct(frames, rotations, translations, calib.getCameraMatrix().clone(), points3D, true);

    cv::Mat frame = video.getNextFrame().value();
    std::optional<cv::Mat> next = video.getNextFrame();
    std::vector<cv::Vec3f> displayPointCloud;
    while(next){
        frame = calib.rectifyImage(frame);
        cv::Mat nextFrame = calib.rectifyImage(next.value());

        auto pointCloud = matching.featureMatching(frame, nextFrame);
        auto rotation = matching.getRotation();
        auto translation = matching.getTranslation();

        std::vector<cv::Vec3b> displayColor;
        for (int i = 0; i < pointCloud.rows; i++){
            auto point = pointCloud.at<cv::Vec3f>(i);
            displayPointCloud.push_back(point);
        }

        int x = int(translation.at<double>(0)) + 300;
        int y = int(translation.at<double>(2)) + 100;
        cv::circle(trajectory, cv::Point(x, y) ,1, CV_RGB(255,0,0), 2);

        objectDetection.detectImage(frame);
        cv::Mat combinedFrame(std::max(frame.rows, trajectory.rows), frame.cols + trajectory.cols, CV_8UC3);
        cv::Mat leftFragment(combinedFrame, cv::Rect(0, 0, frame.size().width, frame.size().height));
        frame.copyTo(leftFragment);
        cv::Mat rightFragment(combinedFrame, cv::Rect(frame.size().width, 0, trajectory.size().width, trajectory.size().height));
        trajectory.copyTo(rightFragment);

        cv::imshow("", combinedFrame);
        cv::waitKey(1);
        frame = next.value();
        next = video.getNextFrame();
    }

    cv::viz::Viz3d window("PCL");
    window.setWindowSize(cv::Size(500, 500));
    window.setWindowPosition(cv::Point(150, 150));

    cv::viz::WCloud cloud(displayPointCloud);
    window.showWidget("123",cloud);
    window.spin();
}