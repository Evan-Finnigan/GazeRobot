#include "opencv2/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/video/tracking.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>

using namespace cv;

using namespace std; 

#define DATA_DIM 2
#define R 2

class pr
{
    int k, n, n_data, num_valid;
    cv::Mat A,b, w, I;
    int factorial(int t) {
        int result = 1;
        for (int i = t; i > 0; i++) {
            result = result * i;
        }
        return result; 
    }
    int choose(int a, int b) {
        return factorial(a) / (factorial(b) * factorial(a - b));
    }
    void buildPoly(float data_point[DATA_DIM], cv::Mat data) {
        data.at<float>(0,0) = 1;
        data.at<float>(0,1) = data_point[0];
        data.at<float>(0,2) = data_point[1];
        data.at<float>(0,3) = data_point[0] * data_point[1];
        data.at<float>(0,4) = data_point[0] * data_point[0];
        data.at<float>(0,5) = data_point[1] * data_point[1];
    }
    public: 
    pr() { 
        num_valid = 25; 
        A = Mat::zeros( 0, DATA_DIM, CV_32FC1);
        b = Mat::zeros( 0, 3, CV_32FC1);
        I = Mat::eye(n, n, CV_32FC1) * 1;
        n = 6;
        w = Mat::zeros(3,n,CV_32FC1);
    } 
    void add(float data_point[DATA_DIM], float x, float y, float z){
        cout << data_point[0] << " " << data_point[1] << endl;
        cv::Mat data_point_mat(1,DATA_DIM, CV_32FC1, data_point); 
        cv::Mat ans = Mat::zeros(1, 3, CV_32FC1);
        cv::Mat data = Mat::zeros(1, n, CV_32FC1);
        ans.at<float>(0,0) = x;
        ans.at<float>(0,1) = y;
        ans.at<float>(0,2) = z;

        buildPoly(data_point, data);

        A.push_back(data);
        b.push_back(ans);

        n_data++;

        if (n_data > n) {
            w = (A.t() * A + I).inv() * A.t() * b;
        }
        if (n_data == num_valid)
            cout << w << endl; 
    }
      
    int eval(float &x, float &y, float &z, float d[DATA_DIM])
    {
        if (n_data > n) {
            cv::Mat data = Mat::zeros(1, n, CV_32FC1); 
            buildPoly(d, data);
            cv::Mat result = data * w;
            x = result.at<float>(0,0);
            y = result.at<float>(0,1);
            z = result.at<float>(0,2);
        }

        if (n_data >= num_valid)
            return 1; 
        return 0; 
    }
};