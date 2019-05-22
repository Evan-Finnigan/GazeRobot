#include <opencv2/opencv.hpp>


#define NUM_LINE_POINTS 5
#define NUM_GRID_POINTS 15

#define PI 3.14159265

class PlanarVisualization{
	private:
		float gridBorderPoints[NUM_GRID_POINTS][2];
		// float linePoints[NUM_LINE_POINTS][2][2];
	public:
		PlanarVisualization(int w, int h, int dist){
			gridBorderPoints[0][0] = w/2;
			gridBorderPoints[0][1] = h;

			int i = 0;
			for(int d = 1; d < 3; d++){
				for(int a = 0; a <= 6; a++){
					gridBorderPoints[i][0] = w/2 + d * dist * cos(a * (PI/6));
					gridBorderPoints[i][1] = h - d * dist * sin(a * (PI/6));
					i++;
				}
			}
		}
		void drawGrid(cv::Mat & img, int hot = -1){
			// Draw the gridlines
			// for(int i = 0; i < NUM_LINE_POINTS; i++){
			// 	cv::line(img, cv::Point2f(linePoints[i][0][0], linePoints[i][0][1]), cv::Point2f(linePoints[i][1][0], linePoints[i][1][1]), cv::Scalar(255, 255, 255));
			// }

			//Draw the grid points
			for(int i = 0; i < NUM_GRID_POINTS; i++){
				if(i == hot)
					cv::circle(img, cv::Point2f(gridBorderPoints[i][0], gridBorderPoints[i][1]), 10, cv::Scalar(0, 255, 0));
				else
					cv::circle(img, cv::Point2f(gridBorderPoints[i][0], gridBorderPoints[i][1]), 5, cv::Scalar(255, 255, 255));
			}
		}
};