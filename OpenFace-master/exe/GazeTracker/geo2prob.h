#include "opencv2/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/video/tracking.hpp"

using namespace cv;

KalmanFilter setup_KF(float x, float y){
    // From http://opencvexamples.blogspot.com/2014/01/kalman-filter-implementation-tracking.html
    KalmanFilter KF(2, 2, 0);
   
    // intialization of KF...

    KF.transitionMatrix = (Mat_<float>(2, 2) << 1,0,   0,1);

    KF.statePre.at<float>(0) = x;
    KF.statePre.at<float>(1) = y;
    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(.1));
    setIdentity(KF.measurementNoiseCov, Scalar::all(10));
    setIdentity(KF.errorCovPost, Scalar::all(.1));

    return KF; 
}


class geo2prob
{
    float x_raw = 0.0;
    float y_raw = 0.0; 
    KalmanFilter KFl;
    KalmanFilter KFr; 
    int KF_initl = 0; 
    int KF_initr = 0;  
     static const int DATA_DIM = 9;
    cv::Mat A = Mat::zeros(3,3, CV_32FC1);
    cv::Mat b = Mat::zeros(3,1, CV_32FC1);
    cv::Mat c = Mat::zeros(3,1, CV_32FC1);

    public: 
    geo2prob(cv::Mat rot, cv::Mat trans, cv::Mat shift) 
    { 
        rot.col(0).copyTo(A.col(0));
        rot.col(2).copyTo(A.col(1));
        cv::Mat wt = rot * (shift) + trans;
        wt.copyTo(c);
    } 
    void add(float data_point[DATA_DIM], float x, float z, int is_l, ofstream *txtOut){
        
    }
    cv::Mat back_calc(float x, float z, cv::Point3f p) {
        b.at<float>(0,0) = c.at<float>(0,0) - p.x;
        b.at<float>(1,0) = c.at<float>(1,0) - p.y;
        b.at<float>(2,0) = c.at<float>(2,0) - p.z;

        cv::Mat d_data = A.col(0) * x + A.col(1) * z + b;
        return d_data / cv::norm(d_data);
    }
    void project(cv::Mat &x, cv::Mat &P, cv::Point3f e, cv::Point3f p, int &KF_init, KalmanFilter &KF, int is_l, ofstream *txtOut){
        
        A.at<float>(0,2) = -e.x;
        A.at<float>(1,2) = -e.y;
        A.at<float>(2,2) = -e.z; 
        b.at<float>(0,0) = p.x - c.at<float>(0,0);
        b.at<float>(1,0) = p.y - c.at<float>(1,0);
        b.at<float>(2,0) = p.z - c.at<float>(2,0); 

        cv::Mat raw = A.inv() * b;

        //Initialize the filter if needed
        if (KF_init == 0){
            KF = setup_KF(raw.at<float>(0,0), raw.at<float>(1,0));
            KF_init = 1;
        }

        // //Step the kalman filter 
        // //From http://opencvexamples.blogspot.com/2014/01/kalman-filter-implementation-tracking.html
        Mat prediction = KF.predict(); 
         Mat_<float> measurement(2,1); measurement.setTo(Scalar(0));
        measurement(0) = raw.at<float>(0,0);
        measurement(1) = raw.at<float>(1,0);
        KF.correct(measurement).copyTo(x);
        KF.errorCovPost.copyTo(P);
    }
      
    int eval(float &x, float &y, float &z, float &x_raw, float &y_raw, float &z_raw, cv::Point3f el, cv::Point3f pl, cv::Point3f er, cv::Point3f pr, cv::Vec6d pe, ofstream *txtOut){ 
        cv::Mat xl = Mat::zeros(2,1, CV_32FC1);
        cv::Mat xr = Mat::zeros(2,1, CV_32FC1);
        cv::Mat Pl = Mat::zeros(2,2, CV_32FC1);
        cv::Mat Pr = Mat::zeros(2,2, CV_32FC1);
        project(xl,Pl,el,pl,KF_initl, KFl, 1, txtOut);
        project(xr,Pr,er,pr,KF_initr, KFr, 0, txtOut);

 
        cv::Mat point = (Pl.inv() + Pr.inv()).inv() * (Pl.inv() * xl + Pr.inv() * xr);
        x = point.at<float>(0,0);
        z = point.at<float>(1,0); 
        if (txtOut != NULL)
            (*txtOut) << x << "," << z << endl;
            // (*txtOut) << x << "," << y << "," << el.x << "," << el.y << "," << el.z << ","
            // << pl.x << "," << pl.y << "," << pl.z << ","
            // << er.x << "," << er.y << "," << er.z << ","
            // << pr.x << "," << pr.y << "," << pr.z << ","
            // << pe[3] << "," << pe[4] << "," << pe[5] << endl;
        return 1;
    }
};