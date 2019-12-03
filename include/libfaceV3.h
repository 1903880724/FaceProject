#ifdef FaceLibDll  
#define LibAPI _declspec(dllexport)  
#else  
#define LibAPI  _declspec(dllimport)  
#endif 
#include "opencv2/opencv.hpp"
#include "mxnet-cpp/MxNetCpp.h"
#include "mxnet/tuple.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#define FACE_TOPLIMIT 50
#define FACE_DEBUG_INFO 1
typedef struct LibAPI FaceInfo {
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	float landmarks[10];
};

enum LibAPI expression
{
	EX_Happy = 0,
	EX_Sad = 1,
	EX_Fear = 2,
	EX_Angry = 3,
	EX_Disgust = 4,
	EX_Surprised = 5,
	EX_Neutral = 6
};//表情枚举

typedef struct LibAPI tagExpressionRecognitionResult
{
	int			_iPosX;		//坐标x
	int			_iPosY;		//坐标y
	int			_iWidth;		//宽度
	int			_iHeight;		//高度
	unsigned int _expression;	//表情枚举
}ExpressionRecognitionResult, *LPExpressionRecognitionResult;

typedef struct LibAPI tagExpressionRecognitionResultSet
{
	ExpressionRecognitionResult _listExpressionRecognitionReulst[FACE_TOPLIMIT];//结果数组
	unsigned int _nListCount;				//数组长度
	unsigned int _nBlockingTime;				//阻塞时间（毫秒）
}ExpressionRecognitionResultSet, *LPExpressionRecognitionResultSet;

typedef struct LibAPI tagExpressionRecognitionConstruction
{
	cv::Mat* _buf;				//图片数据
	unsigned int _size;        //图片数量
	ExpressionRecognitionResultSet _resultSet;			//结果集
}ExpressionRecognitionConstruction, *LPExpressionRecognitionConstruction;

class LibAPI Detector {
public:
	Detector(std::string json_path, std::string params_path, bool use_gpu = 0, int camera_h = 1080, int camera_w = 1920, int process_h = 540, int process_w = 960);
	~Detector();
	void detect(cv::Mat &image, std::vector<FaceInfo>&faces, float scoreThresh = 0.5, float nmsThresh = 0.3);
private:
	void nms(std::vector<FaceInfo>& input, std::vector<FaceInfo>& output, float nmsthreshold = 0.3);
	void decode(mxnet::cpp::NDArray &heatmap, mxnet::cpp::NDArray &scale, mxnet::cpp::NDArray &offset, mxnet::cpp::NDArray &landmarks, std::vector<FaceInfo>&faces, float scoreThresh, float nmsThresh);
	std::vector<int> getIds(const float *heatmap, int h, int w, float thresh);
	void squareBox(std::vector<FaceInfo> &faces);

private:
	float d_scale_h;
	float d_scale_w;

	float scale_h;
	float scale_w;

	int input_h;
	int input_w;

	int image_h;
	int image_w;

	mxnet::cpp::Executor *executor;
	mxnet::cpp::Shape input_shape;
	mxnet::cpp::Shape heatmap_shape;
	mxnet::cpp::Shape scale_shape;
	mxnet::cpp::Shape offset_shape;
	mxnet::cpp::Shape landmarks_shape;
};

class Exp {
public:
	Exp(std::string json_path, std::string params_path, bool use_gpu = 0, int batch_size = 10);
	~Exp();
	std::vector<int> recognise(float* data_faces, int face_num);

private:
	mxnet::cpp::Executor *executor;
	mxnet::cpp::Shape input_shape;
	mxnet::cpp::Shape output_shape;
};

void getFaceFromImage(cv::Mat &input, FaceInfo face_info, float* face);
