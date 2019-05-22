#include <opencv2/core/core.hpp>
#include <opencv2/ml.hpp>

class em
{ 
    cv::Mat eye_data, head_data, labels;
    cv::Ptr<cv::ml::EM> em_model, em_model_temp;
    float table_height;
    
    public: 
    em(int k) 
    { 
        em_model = cv::ml::EM::create();
        em_model->setClustersNumber(k);
        em_model->setCovarianceMatrixType(cv::ml::EM::COV_MAT_SPHERICAL);
        em_model->setTermCriteria(cvTermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 300, 0.1));

        em_model_temp = cv::ml::EM::create();
        em_model_temp->setClustersNumber(k);
        em_model_temp->setCovarianceMatrixType(cv::ml::EM::COV_MAT_SPHERICAL);
        em_model_temp->setTermCriteria(cvTermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 300, 0.1));
    } 
    void add(cv::Mat new_eye_point, cv::Mat new_head_point){
        eye_data.push_back(new_eye_point);
        head_data.push_back(new_head_point);
    }
    void cluster(){
        float min = -1;
        cv::Mat head_pos_shifted, gaze_point, temp_centers, gaze_point_temp, log_probs;
        float compactness;

        cout << eye_data << endl;

        em_model_temp->trainEM(eye_data, log_probs, labels, cv::noArray());

        // head_pos_shifted = head_data;
        // gaze_point = head_data.clone();

        // for(float i = 0; i < 100; i += 10){
        //     head_pos_shifted.col(1) += i;
        //     // need to modify each column of gaze point
        //     gaze_point.col(0) = head_pos_shifted.col(0) + cv::abs(head_pos_shifted.col(1).mul(1/eye_data.col(1))).mul(eye_data.col(0));
        //     gaze_point.col(1) = head_pos_shifted.col(1) + cv::abs(head_pos_shifted.col(1).mul(1/eye_data.col(1))).mul(eye_data.col(1));
        //     gaze_point.col(2) = head_pos_shifted.col(2) + cv::abs(head_pos_shifted.col(1).mul(1/eye_data.col(1))).mul(eye_data.col(2));

        //     em_model_temp->trainEM(gaze_point, log_probs, labels, cv::noArray());

        //     compactness = - static_cast<float>(cv::sum(log_probs)[0]);
        //     cout << compactness << endl;
        //     // compactness = 0;

        //     cout << gaze_point << endl; 

        //     if(compactness < min || min == -1){
        //         table_height = i; 
        //         min = compactness;
        //         em_model->trainEM(gaze_point, log_probs, labels, cv::noArray());
        //         // cout << em_model->getMeans() << endl;
        //     }

        //     head_pos_shifted.col(1) -= i;

        // }

        eye_data.release(); 
        head_data.release();
    }
      
    int eval(cv::Mat eye_point, cv::Mat head_point)
    {
        cv::Mat head_pos_shifted = head_point;
        // head_pos_shifted.col(1) += table_height;
        // cv::Mat gaze_point = head_pos_shifted + abs(head_pos_shifted.at<double>(0, 1)/eye_point.at<double>(0, 1)) * (eye_point);

        // return em_model->predict(gaze_point, cv::noArray());
        cout << eye_data << endl;
        return em_model -> predict(eye_point, cv::noArray());
    }

}; 