#include "opencv2/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/video/tracking.hpp"
#include "GazeCamera.h"
#include <string>
#include "PlanarVisualization.h"

using namespace cv;

class datasaver
{
    string user;
    int point;
    float x;
    float y;
    int i;
    ofstream gazes, poses;
    int collecting = 0; 
    geo2prob * g2;
    GazeCamera * cam; 

    public: 
    datasaver(GazeCamera * cam, vector<string> arguments, PlanarVisualization * pv, int collecting, geo2prob * g2) 
    { 
        this->collecting = collecting;
        if (collecting == 1) {
            this->user = arguments.back();
            arguments.pop_back();
            string point_str = arguments.back();
            arguments.pop_back();
            cout << point_str << endl; 
            this->point = stoi(point_str);
            pv->getPoint(x, y, cam->character_press - '0');
            x = 460 - x;
            y = 500 - y;


            ostringstream osg;
            osg << "dataset" << point << "/gazes.txt";
            string sg = osg.str();
            gazes.open (sg);
            ostringstream osp;
            osp << "dataset" << point << "/poses.txt";
            string sp = osp.str();
            poses.open (sp);

            this->g2 = g2; 
            this->cam = cam; 
        }
    } 
    void cycle()
    {
        if (collecting == 1 && cam->successful_step == true) {
            cv::Mat dl = g2->back_calc(x, y, cam->eyeballCentreLeft);
            cv::Mat dr = g2->back_calc(x, y, cam->eyeballCentreRight);

            ostringstream osdl;
            osdl << "dataset" << point << "/images/" << i << ".jpg";
            string sdl = osdl.str();
            i++;
            ostringstream osdr;
            osdr << "dataset" << point << "/images/" << i << ".jpg";
            string sdr = osdr.str();
            imwrite( sdl, cam->left_eye_patch);
            imwrite( sdr, cam->right_eye_patch);

            i++;

            gazes << dl.at<float>(0,0) << "," << dl.at<float>(1,0) << "," << dl.at<float>(2,0) << endl;
            gazes << dr.at<float>(0,0) << "," << dr.at<float>(1,0) << "," << dr.at<float>(2,0) << endl;

            poses << cam->pose_estimate(3) << "," << cam->pose_estimate(4) << "," << cam->pose_estimate(5) << endl;
            poses << cam->pose_estimate(3) << "," << cam->pose_estimate(4) << "," << cam->pose_estimate(5) << endl;  
        }
    }
    void close() {
        poses.close();
        gazes.close();
    }
};