#include "opencv2/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"

#define DATA_DIM 6

using namespace cv;
using namespace cv::ml;


class svm
{ 
    cv::Mat labels, data;
    Ptr<ml::SVM> s;
    Mat training_data_mean[DATA_DIM];
    Mat training_data_std_dev[DATA_DIM];

    public: 
    svm() 
    { 
        s = ml::SVM::create();
        s->setType(ml::SVM::C_SVC);
        s->setKernel(ml::SVM::POLY);
        s->setGamma(3);
        s->setDegree(1);

    } 
    void add(float data_point[DATA_DIM], int label){
        cv::Mat data_point_mat(1,DATA_DIM, CV_32FC1, data_point);  
        data.push_back(data_point_mat);
        Mat labelMat(1, 1, CV_32SC1, {static_cast<float>(label)});
        labels.push_back(labelMat);
    }
    void cluster(){
        for(int i = 0; i < DATA_DIM; i++){
            meanStdDev( data.col(i), training_data_mean[i], training_data_std_dev[i]);
            data.col(i) = (data.col(i) - training_data_mean[i]) / training_data_std_dev[i];
        }

        s->train(data, ml::ROW_SAMPLE, labels);   
        data.release(); 
        labels.release();

        //down cast to float 32 to prevent type problems in eval
        for(int i = 0; i < DATA_DIM; i++){
            training_data_mean[i].convertTo(training_data_mean[i], CV_32FC1);
            training_data_std_dev[i].convertTo(training_data_std_dev[i], CV_32FC1);
        }
    }
      
    int eval(float data_point[DATA_DIM])
    {
        cv::Mat data_point_mat(1,DATA_DIM, CV_32FC1, data_point); 

        for(int i = 0; i < DATA_DIM; i++){
            data_point_mat.col(i) = (data_point_mat.col(i) - training_data_mean[i]) / training_data_std_dev[i];
        }

        data_point_mat.convertTo(data_point_mat, CV_32FC1);
        return s->predict(data_point_mat);
    }

    void show_data(){
        cout << data << endl;
        cout << labels << endl;
    }

}; 