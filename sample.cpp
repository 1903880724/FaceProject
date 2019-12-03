#include "libfaceV3.h"
#include<time.h>
using namespace cv;
int main()
{
	int batchSize = 20;
	std::string json_path = "D:/LearningFile/masterbusiness/FaceProject2019/centerface-master/models/centerface.json";
	std::string params_path = "D:/LearningFile/masterbusiness/FaceProject2019/centerface-master/models/centerface.params";
	std::string exp_path = "D:/LearningFile/masterbusiness/FaceProject2019/centerface-master/models/fer.json";
	std::string exp_params = "D:/LearningFile/masterbusiness/FaceProject2019/centerface-master/models/fer.params";
	Detector faceDetector(json_path, params_path, 1, 1080, 1920, 540, 960);
	Exp expRecognizer(exp_path, exp_params, 1, batchSize);
	//cv::Mat img = cv::imread("D:/LearningFile/masterbusiness/FaceProject2019/centerface-master/prj-python/340.jpg");
	std::string video = "D:/LearningFile/masterbusiness/FaceProject2019/class.mp4";
	std::string emotion[8] = { "neutral","happy","surprise","sad","neutral","happy","fear","contempt" };
	VideoCapture cap(video, CAP_FFMPEG);
	Mat frame;
	Mat grayframe;
	
	Mat blank = Mat::zeros(64, 64, CV_32FC1);
	//std::vector<Mat> faceList;
	float* data_faces = new float[batchSize * 16384];
	float* face = new float[16384];
	clock_t start[3];
	clock_t end[3];
	while (true) 
	{
		cap.read(frame);
		if (frame.empty()) break;
		cvtColor(frame, grayframe, cv::COLOR_BGR2GRAY);

		std::vector<FaceInfo> face_info;
		std::vector<int> expResult;

		start[0] = clock();
		faceDetector.detect(frame, face_info);
		end[0] = clock();

		start[1] = clock();
		for (int i = 0; i < face_info.size(); ++i)
		{
			//Mat face = getFaceFromImage(frame, face_info[i]);
			getFaceFromImage(frame, face_info[i], face);
			memcpy(data_faces + i * 4096, face, 16384);
			//faceList.push_back(face);
			if (i>0 && (i+1) % batchSize == 0)
			{
				std::vector<int> tmpExp = expRecognizer.recognise(data_faces,batchSize);
				expResult.insert(expResult.end(), tmpExp.begin(), tmpExp.end());
				//std::vector<Mat>().swap(faceList);
			}
			else if (i == face_info.size() - 1)
			{
				int j = i;
				while (j < batchSize)
				{
					j++;
					memcpy(data_faces + j * 4096, blank.data, 16384);
					//faceList.push_back(blank);
				}
				start[2] = clock();
				std::vector<int> tmpExp = expRecognizer.recognise(data_faces,face_info.size()%batchSize);
				end[2] = clock();
				expResult.insert(expResult.end(), tmpExp.begin(), tmpExp.end());
				//std::vector<Mat>().swap(faceList);
			}
			//int tmpExp = expRecognizer.recognize(face);
			//cv::rectangle(frame, cv::Point(face_info[i].x1, face_info[i].y1), cv::Point(face_info[i].x2, face_info[i].y2), cv::Scalar(0, 255, 0), 2);
			//cv::putText(frame, emotion[tmpExp],cv::Point(face_info[i].x1,face_info[i].y1),cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0, 0, 255), 2, 8, 0);
		}
		end[1] = clock();
		//std::cout << "detect time is "<<end[0] - start[0] << std::endl;
		//std::cout << "exp time is "<<end[1] - start[1] << std::endl;
		std::cout<<"real exp time is" << end[2] - start[2] << std::endl;

		for (int i = 0; i < face_info.size(); ++i)
		{
			rectangle(frame, Point(face_info[i].x1, face_info[i].y1), Point(face_info[i].x2, face_info[i].y2), Scalar(0, 255, 0), 2);
			putText(frame, emotion[expResult[i]], Point(face_info[i].x1, face_info[i].y1), FONT_HERSHEY_PLAIN, 3, Scalar(0, 0, 255), 2, 8, 0);
		}

		imshow("camera", frame);
		waitKey(10);

	}
	delete[] data_faces;
	delete[] face;
	return 0;
}


