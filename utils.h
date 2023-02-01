#pragma once
#include <algorithm> 
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <cmath>
#include <numeric> // std::iota 
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include "logging.h"
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

using  namespace cv;
using namespace nvinfer1;


// stuff we know about the network and the input/output blobs
static const int INPUT_H = 640;
static const int INPUT_W = 640;
static const int _segWidth = 160;
static const int _segHeight = 160;
static const int _segChannels = 32;
static const int CLASSES = 80;
static const int Num_box = 8400;
static const int OUTPUT_SIZE = Num_box * (CLASSES+4 + _segChannels);//output0
static const int OUTPUT_SIZE1 = _segChannels * _segWidth * _segHeight ;//output1

static const float CONF_THRESHOLD = 0.1;
static const float NMS_THRESHOLD = 0.5;
static const float MASK_THRESHOLD = 0.5;

const char* INPUT_BLOB_NAME = "images";
const char* OUTPUT_BLOB_NAME = "output0";//detect
const char* OUTPUT_BLOB_NAME1 = "output1";//mask


static Logger gLogger;


std::string keys =
    "{ help  h     | | Print help message. }"
    "{ model       | | Path to input model. }"
    "{ device      |  0 | camera device number. }"
    "{ input      | | Path to input image or video file. Skip this argument to capture frames from a camera. }";


#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)


struct OutputSeg {
    int id;             //id
    float confidence;   //probe
    cv::Rect box;       //box
    cv::Mat boxMask;       //mask
};


static inline cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h, std::vector<int>& padsize) {
	int w, h, x, y;
	float r_w = input_w / (img.cols*1.0);
	float r_h = input_h / (img.rows*1.0);
	if (r_h > r_w) {
		w = input_w;
		h = r_w * img.rows;
		x = 0;
		y = (input_h - h) / 2;
	}
	else {
		w = r_h * img.cols;
		h = input_h;
		x = (input_w - w) / 2;
		y = 0;
	}
	cv::Mat re(h, w, CV_8UC3);
	cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
	cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
	re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
	padsize.push_back(h);
	padsize.push_back(w);
	padsize.push_back(y);
	padsize.push_back(x);// int newh = padsize[0], neww = padsize[1], padh = padsize[2], padw = padsize[3];

	return out;
}


void doInference(IExecutionContext& context, float* input, float* output, float* output1, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 3);
    void* buffers[3];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    const int outputIndex1 = engine.getBindingIndex(OUTPUT_BLOB_NAME1);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));//
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * OUTPUT_SIZE1 * sizeof(float)));
    // cudaMalloc分配内存 cudaFree释放内存 cudaMemcpy或 cudaMemcpyAsync 在主机和设备之间传输数据
    // cudaMemcpy cudaMemcpyAsync 显式地阻塞传输 显式地非阻塞传输
    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(output1, buffers[outputIndex1], batchSize * OUTPUT_SIZE1 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    CHECK(cudaFree(buffers[outputIndex1]));
}

void DrawPred_video(Mat& img,std:: vector<OutputSeg> result) {
    //generate random color
    std::vector<Scalar> color;
    srand(time(0));
    for (int i = 0; i < CLASSES; i++) {
        int b = rand() % 256;
        int g = rand() % 256;
        int r = rand() % 256;
        color.push_back(Scalar(b, g, r));
    }
    Mat mask = img.clone();
    for (int i = 0; i < result.size(); i++) {
        int left, top;
        left = result[i].box.x;
        top = result[i].box.y;
//        int color_num = i;
        rectangle(img, result[i].box, color[result[i].id], 2, 8);

        mask(result[i].box).setTo(color[result[i].id], result[i].boxMask);
//        char label[100];
//        std::cout << "result[i].id = " << result[i].id << std::endl;
//        sprintf(label, "%d:%.2f", result[i].id, result[i].confidence);

        std::vector<std::string> className = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush" };

        auto font_face = cv::FONT_HERSHEY_TRIPLEX;
        auto font_scale = 0.8;
        auto thickness = 2;
        int baseline = 2;

        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << result[i].confidence;

        std::string Mystr = className[result[i].id] +  " " + ss.str();

        cv::Size label_size = cv::getTextSize(Mystr, font_face, font_scale, thickness, &baseline);


        cv::putText(img, Mystr, cv::Point(left, top - label_size.height), font_face, font_scale,
                    cv::Scalar(255, 255, 255), thickness);

    }

    cv::addWeighted(img, 0.3, mask, 0.8, 1, img);

}



