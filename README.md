# FaceProject
![image](https://github.com/terrencewayne/FaceProject/blob/master/faceproject.gif "gif")  

This is a Face Project which I construct for lab and enterprise's demand.

This project contains three main functions:

1.Face Detection. 2.Face Recognition. 3.Expression(Emotion) Recognition.

All functions can recieve both image and video input. And achieve real-time with 1080P videos.

## Getting Started

### Prerequisites

CUDA9.2 and cuDNN7.6

OpenCV 3.4.5

MXNet 1.5.1 with C++ API interface

~~tensorRT 5~~ (I reconstruct the deployment and TensorRT is not need any more)

### File Structure

bin: Dlls are put here. Limited by size I upload libfaceV3.dll only. The other dlls come from prerequisites above.

include: Including files are put here. libfaceV3.h is the header file of the project.

lib: Lib files are put here. libfaceV3.lib contains APIs of this project.

models: Model files are put here. It contains face detection models only. I upload the other models with [BaiduYun Drive](https://pan.baidu.com/s/1xwaZNZueB0qiFExgiQqdZw)

src: The sample is put here.
