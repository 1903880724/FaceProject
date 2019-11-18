#pragma once
#include "opencv2/opencv.hpp"
#include "mxnet-cpp/MxNetCpp.h"
#include "mxnet/tuple.h"
#include "tensorRT/NvInfer.h"
#include "common/argsParser.h"
#include "common/common.h"
#include "common/logger.h"
#include "tensorRT/NvCaffeParser.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#define FACE_TOPLIMIT 50
#define FACE_DEBUG_INFO 1

typedef struct FaceInfo {
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	float landmarks[10];
};//用于存放人脸信息的类，(x1,y1)、(x2,y2)分别为人脸框左上、右下坐标，
//score代表确信为人脸的置信度，landmarks代表脸部五个关键点的坐标。

enum expression
{
	EX_Happy = 0,
	EX_Sad = 1,
	EX_Fear = 2,
	EX_Angry = 3,
	EX_Disgust = 4,
	EX_Surprised = 5,
	EX_Neutral = 6
};//表情枚举

typedef struct tagExpressionRecognitionResult
{
	int			_iPosX;		//坐标x
	int			_iPosY;		//坐标y
	int			_iWidth;		//宽度
	int			_iHeight;		//高度
	unsigned int _expression;	//表情枚举
}ExpressionRecognitionResult, *LPExpressionRecognitionResult;

typedef struct tagExpressionRecognitionResultSet
{
	ExpressionRecognitionResult _listExpressionRecognitionReulst[FACE_TOPLIMIT];//结果数组
	unsigned int _nListCount;				//数组长度
	unsigned int _nBlockingTime;				//阻塞时间（毫秒）
}ExpressionRecognitionResultSet, *LPExpressionRecognitionResultSet;

typedef struct tagExpressionRecognitionConstruction
{
	cv::Mat* _buf;				//图片数据
	unsigned int _size;        //图片数量
	ExpressionRecognitionResultSet _resultSet;			//结果集
}ExpressionRecognitionConstruction, *LPExpressionRecognitionConstruction;

/*
检测器类型
public：
构造函数Detector()
析构函数~Detector()
检测函数detect()
*/
class Detector {
public:
    /*构造函数参数
	json_path：模型文件.json路径
	params_path：模型文件.params路径
	use_gpu：1使用GPU，0不使用GPU
	camera_h, camera_w：输入图像的高和宽
	process_h, process_w：缩放后的图像的高和宽
	*/
	Detector(std::string json_path, std::string params_path, bool use_gpu = 0, int camera_h = 1080, int camera_w = 1920, int process_h = 540, int process_w = 960);
	~Detector();
	/*
	检测函数参数
	image：输入图片
	faces：存放人脸检测结果的容器
	scoreThresh：人脸置信度的阈值（0,1），大于这个值的才认为是人脸，一般保持默认
	nmsThresh：nms操作的阈值（0,1），低于这个值的人脸框会被过滤，一般保持默认
	*/
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


/*
表情识别器类型
public：
构造函数ExpNet()
析构函数~ExpNet()
识别函数recognize()
*/
class ExpNet
{
	template <typename T>
	using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    /*
	构造函数参数
	dataDir:表情模型文件的父文件夹路径
	symbolName:模型文件.prototxt
	weightName:模型文件.caffemodel
	*/
	ExpNet(std::string dataDir, std::string symbolName, std::string weightName);
	~ExpNet();
	/*
	识别函数参数
	face：待识别的人脸图像，返回代表特定表情的整形
	*/
	int recognize(cv::Mat &face);
	samplesCommon::CaffeSampleParams mParams;

private:
	std::shared_ptr<nvinfer1::ICudaEngine> mEngine = nullptr; //!< The TensorRT engine used to run the network
	nvinfer1::Dims mOutputDims;
	samplesCommon::CaffeSampleParams initializeSampleParams(std::string dataDir, std::string symbolName, std::string weightName);
	bool build();
	bool teardown();
	void constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder, SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvcaffeparser1::ICaffeParser>& parser);
};

/*
此函数根据人脸检测得到的坐标，从输入图像上切割出对应的人脸，用以进一步做表情识别
参数
input：原始的输入图像
face_info：人脸检测结果
*/
cv::Mat getFaceFromImage(cv::Mat &input, FaceInfo face_info);
