#include "opencv2/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"

class complicated_km
{ 
    cv::Mat centers, labels, eye_data, head_data;
    int k, w;
    double table_height;
    float norm(cv::Mat point, int n){
        float ret = 0;
        for (int i=0; i<n; i++){
            ret += point.at<float>(i) * point.at<float>(i); 
        }
        return ret;
    }
    public: 
    complicated_km(int k) 
    { 
        this->k = k;
    } 
    void add(cv::Mat new_eye_point, cv::Mat new_head_point){
        eye_data.push_back(new_eye_point);
        head_data.push_back(new_head_point);
    }
    void cluster(){
        float min = -1;
        cv::Mat head_pos_shifted, gaze_point, temp_centers, gaze_point_temp;
        // cv::Mat points(100, 2, CV_32F);
        float compactness;

        // head_pos_shifted = head_data;
        // gaze_point = head_data.clone();

        compactness = kmeans(eye_data, k, labels, 
                cvTermCriteria( cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 10, 1.0),
                3, cv::KMEANS_PP_CENTERS, centers);

        // for(float i = 0; i < 101; i += 10){
        //     head_pos_shifted.col(1) += i;
        //     // need to modify each column of gaze point
        //     gaze_point.col(0) = head_pos_shifted.col(0) + cv::abs(head_pos_shifted.col(1).mul(1/eye_data.col(1))).mul(eye_data.col(0));
        //     gaze_point.col(1) = head_pos_shifted.col(1) + cv::abs(head_pos_shifted.col(1).mul(1/eye_data.col(1))).mul(eye_data.col(1));
        //     gaze_point.col(2) = head_pos_shifted.col(2) + cv::abs(head_pos_shifted.col(1).mul(1/eye_data.col(1))).mul(eye_data.col(2));

        //     compactness = kmeans(gaze_point, k, labels, 
        //         cvTermCriteria( cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 10, 1.0),
        //         3, cv::KMEANS_PP_CENTERS, temp_centers);

        //     if(compactness < min || min == -1){
        //         min = compactness;
        //         table_height = i;
        //         centers = temp_centers.clone();
        //         gaze_point_temp.release();
        //         gaze_point_temp = gaze_point.clone();
        //         cout << gaze_point_temp << endl;
        //     }

        //     head_pos_shifted.col(1) -= i;

        // }

        // points.col(0) = gaze_point_temp.col(0);
        // points.col(1) = gaze_point_temp.col(2);

        // plotting
        // cv::Mat cluster_img(800, 800, CV_8UC3);
        // cv::Scalar colorTab[] =
        // {
        //     cvScalar(0, 0, 255),
        //     cvScalar(0,255,0),
        //     cvScalar(255,100,100),
        //     cvScalar(255,0,255),
        //     cvScalar(0,255,255)
        // };
        // cluster_img = cv::Scalar::all(0);
        // for(int i = 0; i < 100; i++ )
        // {
        //     int clusterIdx = labels.at<int>(i);
        //     cv::Point ipt = points.at<cv::Point2f>(i) + cv::Point2f(100.0f, 100.0f);
        //     circle( cluster_img, ipt, 2, colorTab[clusterIdx], cv::FILLED, cv::LINE_AA );
        //     cout << ipt << "," << endl;
        // }

        // for (int i = 0; i < 3; ++i)
        // {
        //     cv::Point2f c = centers.at<cv::Point2f>(i) + cv::Point2f(100.0f, 100.0f);
        //     circle( cluster_img, c, 40, colorTab[i], 1, cv::LINE_AA );
        // }
        // for(;;){
        //     cv::imshow("clusters", cluster_img);
        //     char key = (char)cv::waitKey();
        //     if( key == 27 || key == 'q' || key == 'Q' ) // 'ESC'
        //         break;
        // }
        // end plotting

        eye_data.release(); 
        head_data.release();
    }
      
    int eval(cv::Mat eye_point, cv::Mat head_point)
    {
        float min_val = std::numeric_limits<float>::max();
        int min = -1; 
        // cv::Mat head_pos_shifted = head_point;
        // head_pos_shifted.col(1) += table_height;
        // cv::Mat gaze_point = head_pos_shifted + abs(head_pos_shifted.at<double>(0, 1)/eye_point.at<double>(0, 1)) * (eye_point);

        // for(int i=0; i < k; i++){
        //     float curr = norm(gaze_point - centers.row(i), 3);
        //     if (curr < min_val){
        //         min_val = curr; 
        //         min = i; 
        //     }
        // }
        for(int i=0; i < k; i++){
            float curr = norm(eye_point - centers.row(i), 3);
            if (curr < min_val){
                min_val = curr; 
                min = i; 
            }
        }
        return min; 
    }

}; 