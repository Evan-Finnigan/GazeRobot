#include "opencv2/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"

class km
{ 
    cv::Mat centers, labels, eye_data;
    int k, w, dim;
    double table_height;
    float norm(cv::Mat point, int n){
        float ret = 0;
        for (int i=0; i<n; i++){
            ret += point.at<float>(i) * point.at<float>(i); 
        }
        return ret;
    }
    public: 
    km(int k, int dim) 
    { 
        this->k = k;
        this->dim = dim;
    } 
    void add(cv::Mat new_eye_point, cv::Mat new_head_point){
        eye_data.push_back(new_eye_point);
    }
    void cluster(){
        kmeans(eye_data, k, labels, 
                cvTermCriteria( cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 10, 1.0),
                dim, cv::KMEANS_PP_CENTERS, centers);

        eye_data.release(); 
    }
      
    int eval(cv::Mat eye_point, cv::Mat head_point)
    {
        float min_val = std::numeric_limits<float>::max();
        int min = -1; 
        for(int i=0; i < k; i++){
            float curr = norm(eye_point - centers.row(i), dim);
            if (curr < min_val){
                min_val = curr; 
                min = i; 
            }
        }
        return min; 
    }

}; 