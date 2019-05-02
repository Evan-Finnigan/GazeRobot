#ifndef __GAZE_CAMERA_h_
#define __GAZE_CAMERA_h_

#include "LandmarkCoreIncludes.h"
#include "GazeEstimation.h"
#include "RotationHelpers.h"

#include "findEyeCenter.h"
#include "findEyeCorner.h"
#include "constants.h"

#include <SequenceCapture.h>
#include <Visualizer.h>
#include <VisualizationUtils.h>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
// #include <torch/script.h>

/**
*      This file is intended to simplify the process of getting the data from images. It is used in the 
*	   following way.
*      
*      GazeCamera cam1(0);
*      while(cam1.step() == true){
*	       if(cam1.successful_step){
*		       Point3f adjustedEyeballCentre0 = cam1.eyeballCentre0;
*              // use data here 
*		   }
*	   }
**/

class GazeCamera {
	private:
		LandmarkDetector::FaceModelParameters * det_parameters;
		LandmarkDetector::CLNF * face_model;
		Utilities::SequenceCapture sequence_reader;
		Utilities::Visualizer * visualizer;
		cv::VideoWriter * video;
		Utilities::FpsTracker fps_tracker;
		cv::Mat_<uchar> grayscale_image;
		int device_number;
		std::vector<cv::Point2f> offsets;
		int eyeImgNum;

		std::ofstream dataFile;

		cv::Point2f smallest_point(cv::Point corners[6]);
		cv::Point2f largest_point(cv::Point corners[6]);
		cv::Mat extractEyeRegion(cv::Point corners[6], cv::Mat in_image, int num_points);
		cv::Rect findEyeRect(cv::Point corners[6], cv::Mat in_image);
		cv::Mat getNormalizedEye(cv::Mat image, cv::Point3f eyeballCentreLeft, cv::Point3f eyeballCentreRight, bool left);

	public:
		//values that will be recorderd every time we step the GazaCamera
		cv::Point3f gazeDirectionLeft;
		cv::Point3f gazeDirectionRight;
		cv::Point3f eyeballCentreLeft;
		cv::Point3f eyeballCentreRight;
		// these variables are for collecting data and there should be an abstraction barrier here
		// when this is finished
		cv::Point2f pupilCentreRight;
		cv::Point2f pupilCentreLeft;
		cv::Vec6d pose_estimate;
		bool successful_step;
		cv::Mat left_eye_patch;
		cv::Mat right_eye_patch;
		cv::Mat rgb_image;
		bool cont;
		char character_press;

		GazeCamera(int device_number);
		GazeCamera(vector<string> arguments);

		bool step();

		void reset();

		~GazeCamera();
};

#endif
