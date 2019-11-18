#include "libfaceV3.h"

int main()
{
	std::string json_path = "centerface.json";//检测模型.json的路径
	std::string params_path = "centerface.params";//检测模型.params的路径
	std::string exp_path = "models/";//表情模型父文件夹的路径
	Detector faceDetector(json_path, params_path, 1, 1080, 1920, 540, 960);//建立检测器
	ExpNet expRecognizer(exp_path, "test_vgg.prototxt", "_iter_5000.caffemodel");//建立识别器

	std::string video = "D:/LearningFile/masterbusiness/FaceProject2019/0_clip.avi";
	std::string emotion[7] = { "happy","sad","fear","angry","disgust","neutral","neutral" };//表情共6种
	cv::VideoCapture cap(video, cv::CAP_FFMPEG);
	cv::Mat frame;
	while (true) 
	{
		cap.read(frame);
		if (frame.empty()) break;
		std::vector<FaceInfo> face_info;//声明人脸检测结果变量
		faceDetector.detect(frame, face_info);//人脸检测
		for (int i = 0; i < face_info.size(); i++)
		{
			cv::Mat face = getFaceFromImage(frame, face_info[i]);//根据检测结果，从原图切割出人脸图
			int tmpExp = expRecognizer.recognize(face);//表情识别
			cv::rectangle(frame, cv::Point(face_info[i].x1, face_info[i].y1), cv::Point(face_info[i].x2, face_info[i].y2), cv::Scalar(0, 255, 0), 2);//画人脸框
			cv::putText(frame, emotion[tmpExp],cv::Point(face_info[i].x1,face_info[i].y1),cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0, 0, 255), 2, 8, 0);//画表情信息
		}
		cv::imshow("camera", frame);//显示结果帧
		cv::waitKey(10);

	}
	system("pause");
	return 0;
}


