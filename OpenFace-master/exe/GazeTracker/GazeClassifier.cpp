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
#include "lwlr.h"
#include "pr.h"
#include "datasaver.h"
#include <time.h>
#include <sstream>
#include <fstream>

// #include <boost/asio/serial_port.hpp> 

 
// using namespace boost;

// using namespace cv;

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
cv::Mat rot =  (cv::Mat_<float>(3,3) << .99, -.008, -.009, -.012, -.93, -.34, -.0061, -.35, -.93);
cv::Mat trans =  (cv::Mat_<float>(3,1) << -100,200,530);
cv::Mat shift =  (cv::Mat_<float>(3,1) << 120,-120,250);
geo2prob g2 = geo2prob(rot, trans, shift);
lwlr lw = lwlr(3);
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
	int collecting = 0; 

	vector<string> arguments = get_arguments(argc, argv);

	if (arguments.size() == 5 || arguments.size() == 4)
		collecting = 1;  
  
	GazeCamera cam1(arguments);

	Mat out_image =  Mat::zeros( h, w, CV_8UC3 );
	namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.  
	setMouseCallback("Display window", cbmouse, data);
	// time_t t = time(NULL);
	// struct tm tm = *localtime(&t);
	// ofstream txtOut;
	// ostringstream os;
	// os << "data/data_" << tm.tm_mon + 1 << "-" << tm.tm_mday << "_" << tm.tm_hour << ":" << tm.tm_min << ".txt";
	// string s = os.str();
	// txtOut.open (s);
	int i = 0; 
	int steps = 50; 
	int goalx = -225;
	int goaly = 100;
	// ostringstream osl;
	// ostringstream osr; 
	// osl << "data/data_l" << tm.tm_mon + 1 << "-" << tm.tm_mday << "_" << tm.tm_hour << ":" << tm.tm_min << ".txt";
	// osr << "data/data_r" << tm.tm_mon + 1 << "-" << tm.tm_mday << "_" << tm.tm_hour << ":" << tm.tm_min << ".txt";
	// string sl = osl.str();
	// string sr = osr.str();
	// txtOutl.open (sl);
	// txtOutr.open (sr);


	PlanarVisualization pv(920, 500, 200);

	datasaver ds(&cam1, arguments, &pv, collecting, &g2);

	std::shared_ptr<torch::jit::script::Module> net = torch::jit::load("model.pt");

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
		line(out_image, Point(460 - 225, 500 - 100), Point(460 + 225, 500 - 100), Scalar( 0, 255, 255));
    	line(out_image, Point(460 - 225, 500 - 400), Point(460 + 225, 500 - 400), Scalar( 0, 255, 255));
    	line(out_image, Point(460 + 225, 500 - 100), Point(460 + 225, 500 - 400), Scalar( 0, 255, 255));
    	line(out_image, Point(460 - 225, 500 - 100), Point(460 - 225, 500 - 400), Scalar( 0, 255, 255));
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

		// if (cam1.character_press == 'c') {
		// 	arduino << "C" << endl;
		// 	SP -> WriteData("C", 1):
		// 	cout << "C" << endl;
		// } else if (cam1.character_press == 'e' ) {
		// 	if (x < 300 && x > -300 && z > 0 && z < 250)
		// 		// arduino << "U," << int(x) << "," << int(z) << "," << 5 << "," << .01 << endl;
		// 		cout << "U," << int(x) << "," << int(z) << "," << 5 << "," << .01 << endl;
		// }


		ds.cycle();

		pv.drawGrid(out_image);
		imshow( "Display window", out_image);

		Mat image, greyImage;
		;
		//cv::cvtColor(image, greyImage, CV_BGR2GRAY);
		greyImage = Mat::zeros(36,60, CV_8UC1);
		at::Tensor tensor_image = torch::from_blob(greyImage.data, {1, 1, 36, 60}, at::kByte);
		tensor_image = tensor_image.to(at::kFloat);

		// // // Create a vector of inputs.
		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(tensor_image);

		// // Execute the model and turn its output into a tensor.
		at::Tensor output = net->forward(inputs).toTensor();
		auto output_accessor = output.accessor<float,2>();

		//auto output = net -> forward(inputs).toTuple()->elements()[6].toTensor().clone().clamp(0,255);

		float ex = output_accessor[0][0];
		float ey = output_accessor[0][1];
		float ez = output_accessor[0][2];	

		cout << ex << " " << ey << " " << ez << endl;
	}
	// txtOut.close();
	// txtOutl.close();
	// txtOutr.close();
	ds.close();

	return 0;
}