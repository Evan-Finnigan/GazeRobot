///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt

//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace 2.0: Facial Behavior Analysis Toolkit
//       Tadas Baltrušaitis, Amir Zadeh, Yao Chong Lim, and Louis-Philippe Morency
//       in IEEE International Conference on Automatic Face and Gesture Recognition, 2018  
//
//       Convolutional experts constrained local model for facial landmark detection.
//       A. Zadeh, T. Baltrušaitis, and Louis-Philippe Morency,
//       in Computer Vision and Pattern Recognition Workshops, 2017.    
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-specific normalisation for automatic Action Unit detection
//       Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
///////////////////////////////////////////////////////////////////////////////
// FaceTrackingVid.cpp : Defines the entry point for the console application for tracking faces in videos.

// Libraries for landmark detection (includes CLNF and CLM modules)
#include "LandmarkCoreIncludes.h"
#include "GazeEstimation.h"
#include "RotationHelpers.h"
#include "PlanarVisualization.h"
#include "GazeCamera.h"

#include <SequenceCapture.h>
#include <Visualizer.h>
#include <VisualizationUtils.h>
// #include <math.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <fstream>
#include "svm.h"
#include <string>
// #include <opencv2/opencv.hpp>
// #include <boost/filesystem.hpp>
#include "arduino-serial-lib.hpp"

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

#define NUM_TARGETS 4
#define TRAINING_STEPS 50

//mutex for the continue character
std::mutex mtx;

static void printErrorAndAbort(const std::string & error)
{
	std::cout << error << std::endl;
	abort();
}

void read_buffer(char * buffer){
	for(;;){
		cin >> *buffer;
	}
}

char read_char_async(char * buffer){
	char ret;
	ret = *buffer;
	*buffer = '\0';
	return ret;
}

// adapted from: https://stackoverflow.com/questions/27981214/opencv-how-do-i-multiply-point-and-matrix-cvmat
cv::Point3f operator*(cv::Mat M, const cv::Point3f& p)
{ 
    cv::Mat_<double> src(3/*rows*/,1 /* cols */); 

    src(0,0)=p.x; 
    src(1,0)=p.y; 
    src(2,0)=p.z; 

    cv::Mat_<double> dst = M*src; //USE MATRIX ALGEBRA 
    return cv::Point3f(dst(0,0),dst(1,0),dst(2,0)); 
} 

// float norm(cv::Point3f x){
// 	return sqrt(x.x * x.x + x.y * x.y + x.z * x.z);
// }

#define FATAL_STREAM( stream ) \
printErrorAndAbort( std::string( "Fatal error: " ) + stream )

using namespace std;

vector<string> get_arguments(int argc, char **argv)
{

	vector<string> arguments;

	for (int i = 0; i < argc; ++i)
	{
		arguments.push_back(string(argv[i]));
	}
	return arguments;
}

int mode(int labels[15]){
	int best_label = 0;
	int best_num_label = 0;
	int num_of_label = 0;

	for(int label = 0; label < NUM_TARGETS; label++){
		for(int j = 0; j < 15; j++){
			if(labels[j] == label){
				num_of_label++;
			}
		}
		if(num_of_label > best_num_label){
			best_num_label = num_of_label;
			best_label = label;
		}
		num_of_label = 0;
	}

	return best_label;
}

int main(int argc, char **argv)
{
	//  explain usage of the program
	cout << "\n\n#######################################################" << endl;
	cout << "\n\n Enter a \'c\' to continue from one phase to next" << endl;
	cout << "There are (# of targets phases) to record data for each target \n then there is a final phase to train and start testing \n\n" << endl;
	cout << "#######################################################\n\n" << endl;
	
	// int collecting = 0; 
	int grad_tracker = -1;

	vector<string> arguments = get_arguments(argc, argv);

	for(int i = 0; i != arguments.size(); i++){
		if(arguments[i].compare("-grad_tracker") == 0)
			grad_tracker = i;
	}
	if(grad_tracker > -1)
		arguments.erase(arguments.begin() + grad_tracker);
  
	GazeCamera cam1(arguments, false, grad_tracker > -1 ? true : false);


	// GazeCamera cam1(0);
	// // GazeCamera cam2(1);

	// float w = 800; 
	// Mat out_image =  Mat::zeros( w, w, CV_8UC3 );
	// namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.  

	// our svm
	svm k = svm();
	int labeling_pos = -1;

	int points_recorded_for_curr_target = TRAINING_STEPS;
	char cont;

	// start the thread to record the the continue character
	std::thread recording_thread(read_buffer, &cont);

	// setup the serial connection
	int fd = serialport_init("/dev/cu.usbmodem1411", 115200);
	if(fd != -1)
		serialport_flush(fd);

	// // read in calibration data
	// String imageFolder = argv[1];
	// FileStorage fs(imageFolder + "calibration_data.yaml", FileStorage::READ);
	// FileNode node = fs["tvecs"];
	// // will just read in the tvec and rvec for the first calibration image
	// Mat tvec_temp;
	// node[0] >> tvec_temp;
	// Point3f tvec(tvec_temp.at<double>(0,0), tvec_temp.at<double>(1,0), tvec_temp.at<double>(2,0));

	// // adjust the translation vector to make the table be lower
	// tvec = tvec + Point3f(0., 349.25, 0.);
	// tvec.x = 0;

	// node = fs["rvecs"];
	// Mat rvec;
	// node[0] >> rvec;

	// node = fs["camera_matrix"];
	// Mat K;
	// node[0] >> K;

	// node = fs["distance_coefficients"];
	// Mat d;
	// node[0] >> d;

	// // invert the Rotation matrix
	// Mat R_inv;
	// Rodrigues(rvec, R_inv);
	// R_inv = R_inv.inv();

	PlanarVisualization pv(920, 500, 200);
	float h = 500;
	float w = 920; 
	Mat out_image =  Mat::zeros( h, w, CV_8UC3 );
	int num_stationary = 0;
	int last_point = 0;

	// // points to create the homography
	// vector<Point2f> dstPoints;
	// dstPoints.push_back(Point2f(30, 0));
	// dstPoints.push_back(Point2f(60, 0));
	// dstPoints.push_back(Point2f(60, 30));
	// dstPoints.push_back(Point2f(30, 30));

	// int positions[9][2] = {{-300, 10}, {-200, 10}, {-100, 10}, {0, 200}, {0, 100}, {0, 10}, {100, 10}, {200, 10}, {300, 10}};
	int positions[4][2] = {{0, 0}, {-300, 10}, {0, 200}, {300, 10}};

	INFO_STREAM("Starting tracking");
	while (true) // this is not a for loop as we might also be reading from a webcam
	{
		// If tracking succeeded and we have an eye model, estimate gaze
		cam1.step();

		// calibrate if command sent to calibrate
		if (cam1.character_press == 'c') {
			string to_send = "C\r\n";
			cout << "calibrate" << endl;
			if(fd == -1)
				cout << "Serial port not open" << endl;
			else
				serialport_write(fd, to_send.c_str());
		}

		if (cam1.successful_step)
		{
			// cout << eyeballCentre0 << endl;
			if(points_recorded_for_curr_target >= TRAINING_STEPS && read_char_async(&cont) == 'c'){
				points_recorded_for_curr_target = 0;
				labeling_pos++;
			}
			if(labeling_pos < NUM_TARGETS && points_recorded_for_curr_target < TRAINING_STEPS){
				points_recorded_for_curr_target++;
				cout << "Look at " << labeling_pos << endl;
			} 


			float data_point[6] = {static_cast<float>(cam1.pose_estimate[0]), 
								   static_cast<float>(cam1.pose_estimate[1]), 
								   static_cast<float>(cam1.pose_estimate[2]),
								   cam1.gazeDirectionLeft.x, 
								   cam1.gazeDirectionLeft.y, 
								   cam1.gazeDirectionLeft.z};			

			if(points_recorded_for_curr_target >= TRAINING_STEPS && read_char_async(&cont) == 'c'){
				points_recorded_for_curr_target = 0;
				labeling_pos++;
			}
			if(labeling_pos < NUM_TARGETS && points_recorded_for_curr_target < TRAINING_STEPS){
				k.add(data_point, labeling_pos);
				cout << "Look at position  " << labeling_pos << ": " << points_recorded_for_curr_target << endl;
				points_recorded_for_curr_target++;
			} else if(labeling_pos == (NUM_TARGETS)){
				k.cluster();
				labeling_pos++;
			} else if(labeling_pos > NUM_TARGETS) {
				cout << "evaluating" << endl;
				int point = k.eval(data_point);
				cout << point << endl;
				out_image =  Mat::zeros( h, w, CV_8UC3 );
				pv.drawGrid(out_image, point);
				imshow("Display window", out_image);

				// if the classifier has stayed at one point for over 10 iterations, go to that point
				if(last_point == point){
					num_stationary++;
				}else{
					num_stationary = 0;
				}

				if(num_stationary == 6){
					int x = positions[point][0];
					int z = positions[point][1];
					string to_send = "U," + to_string(x) + "," + to_string(z) + "," + to_string(5) + "," + "0.1\r\n";
					std::cout << to_send << std::endl; 
					if(fd == -1)
						cout << "Serial port not open" << endl << endl;
					else
						serialport_write(fd, to_send.c_str());
				}

				last_point = point; 
			}
		}
	}

	if(fd != -1){
		serialport_close(fd);
		cout << "Serial port closed" << endl;
	}

	return 0;
}

