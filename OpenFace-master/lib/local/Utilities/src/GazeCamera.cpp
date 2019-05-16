#include "GazeCamera.h"

cv::Point2f GazeCamera::smallest_point(cv::Point corners[6]){
	int small_x = INT_MAX;
	int small_y = INT_MAX;
	for(int i = 0; i < 6; i++){
		if(corners[i].x < small_x)
			small_x = corners[i].x;
		if(corners[i].y < small_y)
			small_y = corners[i].y;
	}

	// std::cout << cv::Point2f(small_x, small_y) << std::endl;

	return cv::Point2f(small_x, small_y);
}

cv::Point2f GazeCamera::largest_point(cv::Point corners[6]){
	int large_x = INT_MIN;
	int large_y = INT_MIN;
	for(int i = 0; i < 6; i++){
		if(corners[i].x > large_x)
			large_x = corners[i].x;
		if(corners[i].y > large_y)
			large_y = corners[i].y;
	}

	return cv::Point2f(large_x, large_y);
}
// this will only work with a region defined by six points, it is just made for extracting 
// an eye from an image
cv::Mat GazeCamera::extractEyeRegion(cv::Point corners[6], cv::Mat in_image, int num_points){
	cv::Point2f small = smallest_point(corners);
	cv::Point2f large = largest_point(corners);

	// small = small - (cv::Point2f(150, 50) - (large - small) / 2);

	// subtract away the smallest value of the region we are cropping to
	for(int i = 0; i < 6; i++){
		corners[i].x -= small.x;
		corners[i].y -= small.y;
	}

	int w = large.x;

	in_image = in_image(cv::Rect(small.x, small.y, 30, 10));

	const cv::Point* corner_list[1] = {corners};

	// cv::Mat mask(in_image.rows, in_image.cols, CV_8UC3, cv::Scalar(0, 0, 0));
	// cv::fillPoly(mask, corner_list, &num_points, 1, cv::Scalar(255, 255, 255), 8);
	// cv::Mat result(in_image.rows, in_image.cols, CV_8UC3, cv::Scalar(255,255,255));
	// cv::Mat mask2(in_image.rows, in_image.cols, CV_8UC1, cv::Scalar(0));
	// cv::fillPoly(mask2, corner_list, &num_points, 1, cv::Scalar(255));
	// cv::bitwise_and(in_image, mask, result, mask2);

	// subtract away the smallest value of the region we are cropping to
	for(int i = 0; i < 6; i++){
		corners[i].x += small.x;
		corners[i].y += small.y;
	}

	cv::Mat result = in_image;

	return result;
};  

// Following two functions intended for use with Fabian Timm's EyeLike
// needs and image size
cv::Rect GazeCamera::findEyeRect(cv::Point corners[6], cv::Mat in_image){
	cv::Point2f small = smallest_point(corners);
	cv::Point2f large = largest_point(corners);

	if(small.x < 0 || small.y < 0 || small.y > in_image.size().height || small.x > in_image.size().width || 
		large.x < 0 || large.y < 0 || large.y > in_image.size().height || large.x > in_image.size().width){
		small = cv::Point2f(0, 0);
		large = cv::Point2f(25, 25);
	}

	return cv::Rect(small, large);
};

float norm(cv::Point3f x){
	return sqrt(x.x * x.x + x.y * x.y + x.z * x.z);
};

cv::Mat GazeCamera::getNormalizedEye(cv::Mat image, cv::Point3f eyeballCentreLeft, cv::Point3f eyeballCentreRight, bool left){
	cv::Mat K = (cv::Mat_<double>(3,3) << sequence_reader.fx, 0, sequence_reader.cx, 0, sequence_reader.fx, sequence_reader.cy, 0, 0, 1.0);
	cv::Point3f e_r = left ? eyeballCentreLeft : eyeballCentreRight;

	cv::Point3f z_c = cv::Point3f(e_r);
	cv::Point3f y_c = z_c.cross(eyeballCentreRight - eyeballCentreLeft);
	cv::Point3f x_c = y_c.cross(z_c);

	//normalize z_c, y_c and x_c
	z_c = z_c / norm(z_c);
	y_c = y_c / norm(y_c);
	x_c = x_c / norm(x_c);

	cv::Mat S = (cv::Mat_<double>(3,3) << 1, 0, 0, 0, 1, 0, 0, 0, 690/norm(e_r));
	cv::Mat M = (cv::Mat_<double>(2,3) << 1, 0, -*face_model->detected_landmarks[37], 0, 1, -*face_model->detected_landmarks[105]);

	cv::Mat R = (cv::Mat_<double>(3,3) << x_c.x, x_c.y, x_c.z, y_c.x, y_c.y, y_c.z, z_c.x, z_c.y, z_c.z);

	cv::Mat K_new = (cv::Mat_<double>(3,3) << 960, 0, 25, 0, 960, 10, 0, 0, 1.0);

	cv::Mat H = K_new * S * R * K.inv();

	cv::Mat warpedImage;

	warpPerspective(image, warpedImage, H, cv::Size(60, 36));

	return warpedImage;
}

// constructor for when you have arguements
GazeCamera::GazeCamera(vector<string> arguments, bool record_video, bool grad_tracker){
	this -> visualizer = new Utilities::Visualizer(true, false, false, false);

	this -> gazeDirectionLeft = cv::Point3f(0, 0, -1);
	this -> gazeDirectionRight = cv::Point3f(0, 0, -1);
	this -> eyeballCentreLeft = cv::Point3f(0, 0, 0);
	this -> eyeballCentreRight = cv::Point3f(0, 0, 0);

	this -> successful_step = true;
	this -> record_video = record_video;
	this -> grad_tracker = grad_tracker;

	det_parameters = new LandmarkDetector::FaceModelParameters(arguments);
	face_model = new LandmarkDetector::CLNF(det_parameters->model_location);

	// The sequence reader chooses what to open based on command line arguments provided
	if (!sequence_reader.Open(arguments))
		std::cout << "failed to open sequence" << std::endl;

	if (!this -> face_model -> loaded_successfully){
			std::cout << "ERROR: Could not load the landmark detector" << std::endl;
	}

	if (!this -> face_model -> eye_model){
			std::cout << "WARNING: no eye model found" << std::endl;
	}

	fps_tracker.AddFrame(); 

	std::cout << "Device or file opened" << std::endl;

	this -> rgb_image = sequence_reader.GetNextFrame();

	cout << "image read" << endl;

	//video recording
	if(this -> record_video)
		this -> video = new cv::VideoWriter("recording.avi",CV_FOURCC('M','J','P','G'),10, cv::Size(640, 480));

	// this -> dataFile.open("box_upper_corner.csv", std::ofstream::out | std::ofstream::app);
}

// constructor for when you just want the webcam referenced by a certain device number
GazeCamera::GazeCamera(int device_number, bool record_video, bool grad_tracker){
	this -> visualizer = new Utilities::Visualizer(true, false, false, false);

	this -> gazeDirectionLeft = cv::Point3f(0, 0, -1);
	this -> gazeDirectionRight = cv::Point3f(0, 0, -1);
	this -> eyeballCentreLeft = cv::Point3f(0, 0, 0);
	this -> eyeballCentreRight = cv::Point3f(0, 0, 0);

	this -> successful_step = true;
	this -> grad_tracker = grad_tracker;
	this -> record_video = record_video;
	this -> device_number = device_number;

	this -> eyeImgNum = 0;

	vector<string> arguments;
	arguments.push_back("-device");
	arguments.push_back(to_string(device_number));

	det_parameters = new LandmarkDetector::FaceModelParameters(arguments);
	face_model = new LandmarkDetector::CLNF(det_parameters->model_location);

	// The sequence reader chooses what to open based on command line arguments provided
	if (!sequence_reader.Open(arguments))
		std::cout << "failed to open sequence" << std::endl;

	if (!this -> face_model -> loaded_successfully){
			std::cout << "ERROR: Could not load the landmark detector" << std::endl;
	}

	if (!this -> face_model -> eye_model){
			std::cout << "WARNING: no eye model found" << std::endl;
	}

	fps_tracker.AddFrame(); 

	std::cout << "Device or file opened" << std::endl;

	this -> rgb_image = sequence_reader.GetNextFrame();

	cout << "image read" << endl;

	//video recording
	if(this -> record_video)
		this -> video = new cv::VideoWriter("recording.avi",CV_FOURCC('M','J','P','G'),10, cv::Size(640, 480));
};

bool GazeCamera::step(){
	if(!rgb_image.empty()){
		this -> grayscale_image = sequence_reader.GetGrayFrame();

		bool detection_success = LandmarkDetector::DetectLandmarksInVideo(this->rgb_image, *(this->face_model), *(this->det_parameters), this->grayscale_image);
		this -> pose_estimate = LandmarkDetector::GetPose(*(this->face_model), this->sequence_reader.fx, this->sequence_reader.fy, this->sequence_reader.cx, this->sequence_reader.cy);

		if(detection_success && this->face_model->eye_model){
			// just need these GazeAnalysis::EstimateGaze to get eyeballCentreRight and eyeballCentreLeft
			// GazeAnalysis::EstimateGaze(*(this->face_model), this -> gazeDirectionRight, this -> eyeballCentreRight, this->sequence_reader.fx, this->sequence_reader.fy, this->sequence_reader.cx, this->sequence_reader.cy, false);
			// GazeAnalysis::EstimateGaze(*(this->face_model), this -> gazeDirectionLeft, this -> eyeballCentreLeft, this->sequence_reader.fx, this->sequence_reader.fy, this->sequence_reader.cx, this->sequence_reader.cy, true);

			// this -> left_eye_patch = getNormalizedEye(this -> rgb_image, this -> eyeballCentreLeft, this -> eyeballCentreRight, true);
			// this -> right_eye_patch = getNormalizedEye(this -> rgb_image, this -> eyeballCentreLeft, this -> eyeballCentreRight, false);

			// imshow("Warped Left Eye", this -> left_eye_patch);
			// imshow("Warped Right Eye", this -> right_eye_patch);

			cv::Point corners[6];
			// crop so just the right eye is visible
			corners[0] = cv::Point2f(*this->face_model->detected_landmarks[42], *this->face_model->detected_landmarks[42 + 68]);
			corners[1] = cv::Point2f(*this->face_model->detected_landmarks[43], *this->face_model->detected_landmarks[43 + 68]);
			corners[2] = cv::Point2f(*this->face_model->detected_landmarks[44], *this->face_model->detected_landmarks[44 + 68]);
			corners[3] = cv::Point2f(*this->face_model->detected_landmarks[45], *this->face_model->detected_landmarks[45 + 68]);
			corners[4] = cv::Point2f(*this->face_model->detected_landmarks[46], *this->face_model->detected_landmarks[46 + 68]);
			corners[5] = cv::Point2f(*this->face_model->detected_landmarks[47], *this->face_model->detected_landmarks[47 + 68]);

			// this -> offsets.push_back(smallest_point(corners));
			
			// this -> right_eye_patch = GazeCamera::extractEyeRegion(corners, rgb_image, 6);
			this -> right_eye_patch = this->rgb_image(GazeCamera::findEyeRect(corners, this -> grayscale_image));
			cv::imshow("right eye", this -> right_eye_patch);
			cv::Point2f right_pupil_center_relative = findEyeCenter(this -> grayscale_image, GazeCamera::findEyeRect(corners, this -> grayscale_image), "right debug");
			cv::Point2f ul_corner_r = smallest_point(corners);
			cv::Point2f right_pupil_center = right_pupil_center_relative + ul_corner_r;
			// cv::circle(this -> rgb_image, right_pupil_center, 3, cv::Scalar(255, 255, 255));

			
			if(this->record_video)
				video -> write(rgb_image);

			if(this -> grad_tracker)
				GazeAnalysis::EstimateGaze(*(this->face_model), right_pupil_center, this -> gazeDirectionRight, this -> eyeballCentreRight, this->sequence_reader.fx, this->sequence_reader.fy, this->sequence_reader.cx, this->sequence_reader.cy, false);
			else
				GazeAnalysis::EstimateGaze(*(this->face_model), this -> gazeDirectionRight, this -> eyeballCentreRight, this->sequence_reader.fx, this->sequence_reader.fy, this->sequence_reader.cx, this->sequence_reader.cy, false);

			// std::cout << this -> gazeDirectionRight << std::endl;

			// crop so just the left eye is visible
			corners[0] = cv::Point2f(*this->face_model->detected_landmarks[36], *this->face_model->detected_landmarks[36 + 68]);
			corners[1] = cv::Point2f(*this->face_model->detected_landmarks[37], *this->face_model->detected_landmarks[37 + 68]);
			corners[2] = cv::Point2f(*this->face_model->detected_landmarks[38], *this->face_model->detected_landmarks[38 + 68]);
			corners[3] = cv::Point2f(*this->face_model->detected_landmarks[39], *this->face_model->detected_landmarks[39 + 68]);
			corners[4] = cv::Point2f(*this->face_model->detected_landmarks[40], *this->face_model->detected_landmarks[40 + 68]);
			corners[5] = cv::Point2f(*this->face_model->detected_landmarks[41], *this->face_model->detected_landmarks[41 + 68]);

			// this -> offsets.push_back(smallest_point(corners));

			// std::cout << corners[0] << std::endl;
			
			// this -> left_eye_patch = GazeCamera::extractEyeRegion(corners, rgb_image, 6);
			this -> left_eye_patch = this->rgb_image(GazeCamera::findEyeRect(corners, this -> grayscale_image));
			cv::imshow("left eye", this -> left_eye_patch);
			cv::Point2f left_pupil_center_relative = findEyeCenter(this -> grayscale_image, GazeCamera::findEyeRect(corners, this -> grayscale_image), "left debug");
			cv::Point2f ul_corner_l = smallest_point(corners);
			cv::Point2f left_pupil_center = left_pupil_center_relative + ul_corner_l;
			// cv::circle(this -> rgb_image, left_pupil_center, 3, cv::Scalar(255, 255, 255));

			if(this -> grad_tracker)
				GazeAnalysis::EstimateGaze(*(this->face_model), left_pupil_center, this -> gazeDirectionLeft, this -> eyeballCentreLeft, this->sequence_reader.fx, this->sequence_reader.fy, this->sequence_reader.cx, this->sequence_reader.cy, true);
			else
				GazeAnalysis::EstimateGaze(*(this->face_model), this -> gazeDirectionLeft, this -> eyeballCentreLeft, this->sequence_reader.fx, this->sequence_reader.fy, this->sequence_reader.cx, this->sequence_reader.cy, true);

			// dataFile << ul_corner_l.x << ","<< ul_corner_l.y << "," << ul_corner_r.x << "," << ul_corner_r.y << endl;
			// cv::imwrite("extracted_eye_boxes/" + to_string(eyeImgNum) + "_l.png", this -> left_eye_patch);
			// cv::imwrite("extracted_eye_boxes/" + to_string(eyeImgNum) + "_r.png", this -> right_eye_patch);

			if(this -> record_video)


			this -> eyeImgNum++;

			this -> successful_step = true;
		}else{
			this -> successful_step = false;
			// this -> offsets.push_back(cv::Point2f(0, 0));
			// this -> offsets.push_back(cv::Point2f(0, 0));
		}

		// Keeping track of FPS
		this -> fps_tracker.AddFrame();

		// Displaying the tracking visualizations
		visualizer -> SetImage(this -> rgb_image, this -> sequence_reader.fx, this -> sequence_reader.fy, this -> sequence_reader.cx, this -> sequence_reader.cy);
		visualizer -> SetObservationLandmarks(this -> face_model -> detected_landmarks, this -> face_model -> detection_certainty, this -> face_model -> GetVisibilities());
		visualizer -> SetObservationPose(this -> pose_estimate, face_model -> detection_certainty);
		visualizer -> SetObservationGaze(this -> gazeDirectionLeft, this -> gazeDirectionRight, LandmarkDetector::CalculateAllEyeLandmarks(*(this -> face_model)), LandmarkDetector::Calculate3DEyeLandmarks(*(this -> face_model), this -> sequence_reader.fx, this -> sequence_reader.fy, this -> sequence_reader.cx, this -> sequence_reader.cy), this -> face_model -> detection_certainty);
		visualizer -> SetFps(fps_tracker.GetFPS());
		// detect key presses (due to pecularities of OpenCV, you can get it when displaying images)
		this -> character_press = visualizer -> ShowObservation(std::to_string(this -> device_number));

		// restart the tracker
		if (character_press == 'r')
		{
			this -> face_model -> Reset();
		}
		// quit the application
		else if (character_press == 'q')
		{
			return false;
		}


		// Grabbing the next frame in the sequence
		this -> rgb_image = this -> sequence_reader.GetNextFrame();
	}

	return true;
};

void GazeCamera::reset(){
	// Reset the model, for the next video
	this -> face_model -> Reset();
	this -> sequence_reader.Close();
};

GazeCamera::~GazeCamera(){
	std::cout << "Gaze Camera Closed" << std::endl;
	delete this -> det_parameters;
	delete this -> face_model;
	if(this->record_video){
		this -> video -> release();
		delete this -> video;
	}

	//save the offsets of the created images to a file
	// std::ofstream offset_file;
	// offset_file.open("offsets.csv");
	// offset_file << "X,Y" << std::endl;

	// for(cv::Point2f pt: offsets){
	// 	offset_file << pt.x << "," << pt.y << std::endl;
	// }

	// offset_file.close();
};	