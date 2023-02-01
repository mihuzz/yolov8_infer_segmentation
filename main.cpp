// based on https://github.com/fish-kong/Yolov8-instance-seg-tensorrt
#include "utils.h"

using namespace std;
using namespace nvinfer1;


int main(int argc, char** argv)
{

    cv::CommandLineParser parser(argc, argv, keys);

//    std::string mym = "/home/mih/QtProjects/onx2trt/yolov8n-seg.engine";

    const std::string modeleFile = parser.get<String>("model");

    parser = CommandLineParser(argc, argv, keys);
    parser.about("Use this script to run Yolov8 segmantation using OpenCV.");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    // create a model using the API directly and serialize it to a stream
    char* trtModelStream{ nullptr }; //char* trtModelStream==nullptr;
    size_t size{ 0 };

    std::ifstream file(modeleFile, std::ios::binary);
    if (file.good()) {
        std::cout << "load engine success" << std::endl;
        file.seekg(0, file.end);
        size = file.tellg();
        //std::cout << "\nfile:" << argv[1] << " size is";
        //std::cout << size << "";

        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);//
        file.read(trtModelStream, size);
        file.close();
    }
    else {
        std::cout << "load engine failed" << std::endl;
        return 1;
    }

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    bool didInitPlugins = initLibNvInferPlugins(nullptr, "");
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    // Run inference
    static float prob[OUTPUT_SIZE];
    static float prob1[OUTPUT_SIZE1];


    cv::Mat frame;

    // Open a video file or an image file or a camera stream.
    VideoCapture cap;
    if (parser.has("input"))
        cap.open(parser.get<String>("input"));
    else
        cap.open(parser.get<int>("device"), cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "ERROR! Unable to open camera\n";
        return -1;
    }


    cv::VideoWriter myvideofile("/home/mih/Видео/yolov8segment.avi", cv::VideoWriter::fourcc('I','4','2','0'), 29, cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)));

    for(;;) {


        cap >> frame;

        if(frame.empty()){

            std::cout << "errror " << std::endl;
            return -1;
        }

        // Subtract mean from image
        static float data[3 * INPUT_H * INPUT_W];
        cv::Mat pr_img0, pr_img;

        std::vector<int> padsize;
        pr_img = preprocess_img(frame, INPUT_H, INPUT_W, padsize);       // Resize
        int newh = padsize[0], neww = padsize[1], padh = padsize[2], padw = padsize[3];
        float ratio_h = (float)frame.rows / newh;
        float ratio_w = (float)frame.cols / neww;
        int i = 0;// [1,3,INPUT_H,INPUT_W]
        //std::cout << "pr_img.step" << pr_img.step << std::endl;
        for (int row = 0; row < INPUT_H; ++row) {
            uchar* uc_pixel = pr_img.data + row * pr_img.step;//pr_img.step=widthx3 就是每一行有width个3通道的值
            for (int col = 0; col < INPUT_W; ++col)
            {

                data[i] = (float)uc_pixel[2] / 255.0;
                data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
                data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.;
                uc_pixel += 3;
                ++i;
            }
        }


        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, prob1, 1);
        auto end = std::chrono::system_clock::now();
        std::cout << "NMS：" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;


        std::vector<int> classIds;//id
        std::vector<float> confidences;//prom of id
        std::vector<cv::Rect> boxes;//box id
        std::vector<cv::Mat> picked_proposals;  //data mask

        // box
        int net_length = CLASSES + 4 + _segChannels;
        cv::Mat out1 = cv::Mat(net_length, Num_box, CV_32F, prob);

        start = std::chrono::system_clock::now();
        for (int i = 0; i < Num_box; i++) {

            cv::Mat scores = out1(Rect(i, 4, 1, CLASSES)).clone();
            Point classIdPoint;
            double max_class_socre;
            minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
            max_class_socre = (float)max_class_socre;
            if (max_class_socre >= CONF_THRESHOLD) {
                cv::Mat temp_proto = out1(Rect(i, 4 + CLASSES, 1, _segChannels)).clone();
                picked_proposals.push_back(temp_proto.t());
                float x = (out1.at<float>(0, i) - padw) * ratio_w;  //cx
                float y = (out1.at<float>(1, i) - padh) * ratio_h;  //cy
                float w = out1.at<float>(2, i) * ratio_w;  //w
                float h = out1.at<float>(3, i) * ratio_h;  //h
                int left = MAX((x - 0.5 * w), 0);
                int top = MAX((y - 0.5 * h), 0);
                int width = (int)w;
                int height = (int)h;
                if (width <= 0 || height <= 0) { continue; }

                classIds.push_back(classIdPoint.y);
                confidences.push_back(max_class_socre);
                boxes.push_back(Rect(left, top, width, height));
            }

        }
        //（NMS）
        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD, nms_result);
        std::vector<cv::Mat> temp_mask_proposals;
        std::vector<OutputSeg> output;
        Rect holeImgRect(0, 0, frame.cols, frame.rows);
        for (int i = 0; i < nms_result.size(); ++i) {
            int idx = nms_result[i];
            OutputSeg result;
            result.id = classIds[idx];
            result.confidence = confidences[idx];
            result.box = boxes[idx]& holeImgRect;
            output.push_back(result);
            temp_mask_proposals.push_back(picked_proposals[idx]);
        }

        // mask
        Mat maskProposals;
        for (int i = 0; i < temp_mask_proposals.size(); ++i)
            maskProposals.push_back(temp_mask_proposals[i]);

        Mat protos = Mat(_segChannels, _segWidth * _segHeight, CV_32F, prob1);
        Mat matmulRes = (maskProposals * protos).t();//n*32 32*25600
        Mat masks = matmulRes.reshape(output.size(), { _segWidth,_segHeight });//n*160*160

        std::vector<Mat> maskChannels;
        cv::split(masks, maskChannels);
        Rect roi(int((float)padw / INPUT_W * _segWidth), int((float)padh / INPUT_H * _segHeight), int(_segWidth - padw / 2), int(_segHeight - padh / 2));
        for (int i = 0; i < output.size(); ++i) {
            Mat dest, mask;
            cv::exp(-maskChannels[i], dest);//sigmoid
            dest = 1.0 / (1.0 + dest);//160*160
            dest = dest(roi);
            resize(dest, mask, cv::Size(frame.cols, frame.rows), INTER_NEAREST);
            //crop
            Rect temp_rect = output[i].box;
            mask = mask(temp_rect) > MASK_THRESHOLD;
            output[i].boxMask = mask;
        }
        end = std::chrono::system_clock::now();
        std::cout << "time of postprocess：" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        DrawPred_video(frame, output);

        myvideofile.write(frame);

        cv::imshow("result", frame);
        if (cv::waitKey(1)==27)
                    break;

    }

    return 0;
}
