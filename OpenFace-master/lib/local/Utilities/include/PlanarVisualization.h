#include <opencv2/opencv.hpp>

#ifndef PLANAR_VISUALIZATION
#define PLANAR_VISUALIZATION
#define NUM_LINE_POINTS 5
#define NUM_GRID_POINTS 21

using namespace cv;

#define PI 3.14159265

class PlanarVisualization{
	private:
		float gridBorderPoints[NUM_GRID_POINTS][2];
		int h,w;
		// float linePoints[NUM_LINE_POINTS][2][2];
	public:
		PlanarVisualization(int w, int h, int dist){
			int i = 0;
			for(int d = 1; d <= 2; d++){
				for(int a = 1; a <= 5; a++){
					gridBorderPoints[i][0] = w/2 - d * dist * cos(a * (PI/6));
					gridBorderPoints[i][1] = h - d * dist * sin(a * (PI/6));
					//cout << gridBorderPoints[i][0] << " " << gridBorderPoints[i][0] << endl; 
					i++;
				}
			}
			this->h = h;
			this->w = w;
		}
		void drawGrid(cv::Mat & img){
			// Draw the gridlines
			// for(int i = 0; i < NUM_LINE_POINTS; i++){
			// 	cv::line(img, cv::Point2f(linePoints[i][0][0], linePoints[i][0][1]), cv::Point2f(linePoints[i][1][0], linePoints[i][1][1]), cv::Scalar(255, 255, 255));
			// }

			//Draw the grid points
			for(int i = 0; i < NUM_GRID_POINTS; i++){
				cv::circle(img, cv::Point2f(gridBorderPoints[i][0], gridBorderPoints[i][1]), 5, cv::Scalar(255, 255, 0));
			}
			for(int i = -300; i <= 300; i+=100) {
				cv::line(img, Point(i + 460, 500 - 10), Point(i + 460, 300 - 10), Scalar( 0, 255, 255));
			}
			for(int i = 10; i <= 210; i+=100) {
				cv::line(img, Point(460 - 300, 500 - i), Point(460 + 300, 500 - i), Scalar( 0, 255, 255));
			}
		}
		void getPoint(float &x, float &y, int i){
			x = gridBorderPoints[i][0];
			y = gridBorderPoints[i][1];
		}
};

#endif