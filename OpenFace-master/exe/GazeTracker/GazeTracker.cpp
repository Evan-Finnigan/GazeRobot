// Libraries for landmark detection (includes CLNF and CLM modules)
#include "LandmarkCoreIncludes.h"
#include "GazeEstimation.h"
#include "RotationHelpers.h"
#include "GazeCamera.h"

#include <SequenceCapture.h>
#include <Visualizer.h>
#include <VisualizationUtils.h>
#include <math.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <fstream>
//#include "PlanarVisualization.h"
#include <string>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include "geo2prob.h"
// #include "geo3.h"
// #include "lle.h"
#include "pr.h"
#include "datasaver.h"
#include <time.h>
#include <sstream>
#include <fstream>
#include "arduino-serial-lib.hpp"

using namespace cv;

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

static void printErrorAndAbort(const std::string & error)
{
	std::cout << error << std::endl;
	abort();
}

#define FATAL_STREAM( stream ) \
printErrorAndAbort( std::string( "Fatal error: " ) + stream )

#define DATASET_SIZE 56 * 21 * 5

using namespace std;

float h = 500;
float w = 920; 
cv::Mat rot =  (cv::Mat_<float>(3,3) <<  .99, .02, -.05,
                                          0, -.89, -.44,
                                          -.053, .44, -.89);

cv::Mat trans =  (cv::Mat_<float>(3,1) <<  -87, 40,  279);
cv::Mat shift =  (cv::Mat_<float>(3,1) << 130,-270,270);
// cv::Mat rot =  (cv::Mat_<float>(3,3) << .99, -.008, -.009, -.012, -.93, -.34, -.0061, -.35, -.93);
// cv::Mat trans =  (cv::Mat_<float>(3,1) << -100,200,530);
// cv::Mat shift =  (cv::Mat_<float>(3,1) << 120,-120,250);
geo2prob g2 = geo2prob(rot, trans, shift);
pr p = pr();


//To read in clicks
float data_l[9];
float data_r[9];
float * data[2] = {data_l, data_r};
float d[2];
void copyToArray(float * data, cv::Point3f p1, cv::Point3f p2, cv::Vec6d p3) {
	data[0] = p1.x;data[1] = p1.y;data[2] = p1.z;
	data[3] = p2.x;data[4] = p2.y;data[5] = p2.z;
	data[6] = p3[3];data[7] = p3[4];data[8] = p3[5];
}
void cbmouse(int event, int x, int y, int flags, void*userdata)
{
   	if (event == EVENT_LBUTTONDOWN)
   	{
   		x = - 460 +  x;
   		y = 500 - y;
   		cout << x << "," << y << endl;
   		p.add(d,x,0,y);
   	}
}

vector<string> get_arguments(int argc, char **argv)
{

	vector<string> arguments;

	for (int i = 0; i < argc; ++i)
	{
		arguments.push_back(string(argv[i]));
	}
	return arguments;
}

int main(int argc, char **argv)
{
	// int collecting = 0; 
	int grad_tracker = -1;

	vector<string> arguments = get_arguments(argc, argv);

	// if (arguments.size() == 5 || arguments.size() == 4)
	// 	collecting = 1;  


	for(int i = 0; i != arguments.size(); i++){
		if(arguments[i].compare("-grad_tracker") == 0)
			grad_tracker = i;
	}
	if(grad_tracker > -1)
		arguments.erase(arguments.begin() + grad_tracker);
  
	GazeCamera cam1(arguments, false, grad_tracker > -1 ? true : false);

	Mat out_image =  Mat::zeros( h, w, CV_8UC3 );
	namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.  
	setMouseCallback("Display window", cbmouse, data);
	int i = 0; 
	int steps = 50; 
	int goalx = -225;
	int goaly = 100;

	PlanarVisualization pv(920, 500, 200);

	// datasaver ds(&cam1, arguments, &pv, collecting, &g2);

	// std::shared_ptr<torch::jit::script::Module> net = torch::jit::losad("model.pt");

	// setup the serial connection
	int fd = serialport_init("/dev/cu.usbmodem1411", 115200);
	if(fd != -1)
		serialport_flush(fd);

	INFO_STREAM("Starting tracking");
	while (cam1.step() == true) // this is not a for loop as we might also be reading from a webcam
	{
		float x,y,z,x_raw,y_raw,z_raw;
		float x3, y3, z3;
		copyToArray(data_l, cam1.gazeDirectionLeft, cam1.eyeballCentreLeft, cam1.pose_estimate);
		copyToArray(data_r, cam1.gazeDirectionRight, cam1.eyeballCentreRight, cam1.pose_estimate);
		g2.eval(x,y,z,x_raw,y_raw,z_raw,cam1.gazeDirectionLeft, cam1.eyeballCentreLeft, cam1.gazeDirectionRight, cam1.eyeballCentreRight, cam1.pose_estimate, NULL);
	
		out_image =  Mat::zeros( h, w, CV_8UC3 );
		ellipse( out_image,
           			Point(w/2 - x, h - z),
           			Size( w/80.0, w/80.0 ),
           			0,
           			0,
           			360,
           			Scalar( 255, 255, 255),
           			2,
           			8);

		// cout << x << " " << z << endl;
	
		// line(out_image, Point(460 - 300, 500 -10), Point(460 + 300, 500-10), Scalar( 0, 255, 255));
  //   	line(out_image, Point(460 - 300, 500 - 200), Point(460 + 300, 500 - 200), Scalar( 0, 255, 255));
  //   	line(out_image, Point(460 + 300, 500 -10), Point(460 + 300, 500 - 200), Scalar( 0, 255, 255));
  //   	line(out_image, Point(460 - 300, 500-10), Point(460 - 300, 500 - 200), Scalar( 0, 255, 255));
		d[0] = x;
		d[1] = z; 
		int valid = p.eval(x,y,z, d);

		if (valid == 1) {
			// txtOut << goalx << "," << goaly << "," << x << "," << z << endl; 
			if (goaly == 100 && goalx <225)
				goalx+=5;
			else if (goalx == 225 && goaly < 400)
				goaly+=5;
			else if (goaly == 400 && goalx > -225)
				goalx-=5;
			else 
				goaly-=5;
		}
		ellipse( out_image,
           			Point(w/2 + goalx, h - goaly),
           			Size(20, 20),
           			0,
           			0,
           			360,
           			Scalar( 0, 0, 255),
           			2,
           			8);

		ellipse( out_image,
           			Point(w/2 + x, h - z),
           			Size(w/80.0, w/80.0),
           			0,
           			0,
           			360,
           			Scalar( 0, 255, 255),
           			2,
           			8);

		if (cam1.character_press == 'c') {
			string to_send = "C\r\n";
			cout << "calibrate" << endl;
			if(fd == -1)
				cout << "Serial port not open" << endl;
			else
				serialport_write(fd, to_send.c_str());
		} else if (cam1.character_press == 'e' ) {
			if (x > 300)
				x = 300;
			else if (x < -300)
				x = -300;
			
			if (z < 10)
				z = 10;
			else if (z > 200)
				z = 200;
			string to_send = "U," + to_string(int(x)) + "," + to_string(int(z)) + "," + to_string(5) + "," + "0.1\r\n";
			cout << to_send << endl; 
			if(fd == -1)
				cout << "Serial port not open" << endl << endl;
			else
				serialport_write(fd, to_send.c_str());
		}


		// ds.cycle();

		pv.drawGrid(out_image);
		imshow( "Display window", out_image);

		// Mat image, greyImage;
		// //cv::cvtColor(image, greyImage, CV_BGR2GRAY);
		// greyImage = Mat::zeros(36,60, CV_8UC1);
		// at::Tensor tensor_image = torch::from_blob(greyImage.data, {1, 1, 36, 60}, at::kByte);
		// tensor_image = tensor_image.to(at::kFloat);

		// // // // Create a vector of inputs.
		// std::vector<torch::jit::IValue> inputs;
		// inputs.push_back(tensor_image);

		// // // Execute the model and turn its output into a tensor.
		// at::Tensor output = net->forward(inputs).toTensor();
		// auto output_accessor = output.accessor<float,2>();

		// //auto output = net -> forward(inputs).toTuple()->elements()[6].toTensor().clone().clamp(0,255);

		// float ex = output_accessor[0][0];
		// float ey = output_accessor[0][1];
		// float ez = output_accessor[0][2];	

		// cout << ex << " " << ey << " " << ez << endl;
	}
	// txtOut.close();
	// txtOutl.close();
	// txtOutr.close();
	// ds.close();

	if(fd != -1){
		serialport_close(fd);
		cout << "Serial port closed" << endl;
	}

	return 0;
}
