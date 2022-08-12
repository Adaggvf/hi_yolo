
#include "../common/common.hpp"
#include "../common/RenderImage.hpp"
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core_c.h>
#include <fstream>
#include <thread>
#include <chrono>
using namespace cv;
using namespace std;
using namespace cv::dnn;


#define DEPTH

// Constructor

// 常数
const float INPUT_WIDTH = 640.0f;
const float INPUT_HEIGHT = 640.0f;
const float SCORE_THRESHOLD = 0.5f;
const float NMS_THRESHOLD = 0.45f;
const float CONFIDENCE_THRESHOLD = 0.45f;

// 文本参数
cv::Mat scaleMat;
cv::Mat depth_all;
const float FONT_SCALE = 0.7f;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;
int X;
int Y;
\
Scalar BLACK = Scalar(0, 0, 0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0, 0, 255);


// 绘制预测的边界框
void draw_label(Mat& input_image, string label, int left, int top)    //绘制标签
{
    // Display the label at the top of the bounding box.
    int baseLine;
    Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = max(top, label_size.height);
    // 左上角
    Point tlc = Point(left, top);
    // 右下角。
    Point brc = Point(left + label_size.width, top + label_size.height + baseLine);
    // 绘制黑色矩形
    rectangle(input_image, tlc, brc, BLACK, FILLED);
    // 将标签放在黑色矩形上。
    putText(input_image, label, Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}

vector<Mat> pre_process(Mat& input_image, Net& net)           //预处理
{
    // 转换为 blob.
    Mat blob;
    blobFromImage(input_image, blob, 1. / 255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

    net.setInput(blob);

    // 前向传播
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    return outputs;
}

Mat post_process(Mat& input_image, vector<Mat>& outputs, const vector<string>& class_name)
{
    // 初始化向量以在展开检测时保存各自的输出。
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;

    // 调整大小的因素
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float* data = (float*)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;
    // 迭代 25200 次检测。
    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        // 丢弃错误检测并继续。
        if (confidence >= CONFIDENCE_THRESHOLD)
        {
            float* classes_scores = data + 5;
            // 创建一个 1x85 Mat 并存储 80 个分组。
            Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
            //获取最佳类别分数的索引。
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // 如果分组分数高于阈值继续。
            if (max_class_score > SCORE_THRESHOLD)
            {
                // 将类 ID 和置信度存储在预定义的相应向量中。

                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                // 中心
                float cx = data[0];
                float cy = data[1];
                //框尺寸.
                float w = data[2];
                float h = data[3];
                //边界框坐标。
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                // 将良好的检测存储在框向量中。
                boxes.push_back(Rect(left, top, width, height));
            }

        }
        //跳到下一列。
        data += 85;
    }

    // 执行非最大抑制并绘制预测。
    vector<int> indices;
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    for (int i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        Rect box = boxes[idx];

        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        // 绘制边界框。
        rectangle(input_image, Point(left, top), Point(left + width, top + height), BLUE, 3 * THICKNESS);

        //获取类名的标签及其置信度。
        string label = format("%.2f", confidences[idx]);
        label = class_name[class_ids[idx]] + ":" + label;
        // 绘制类标签。
        draw_label(input_image, label, left, top);
     //  int pixel_val = scaleMat.at<uchar>(top + height / 2, left + width / 2);
        Y = top + height / 2;
        X = left + width / 2;
        int  ve_stat = depth_all.ptr<uchar>(Y)[X]; //获取深度信息
        cout << class_name[class_ids[idx]] << "位置：" << "Y：" << Y << " X：" << X << " 深度：" << ve_stat <<endl;

    }
    return input_image;
}





int main(int argc, char** argv)
{
    MV3D_RGBD_VERSION_INFO stVersion;
    ASSERT_OK(MV3D_RGBD_GetSDKVersion(&stVersion));
    LOGD("dll version: %d.%d.%d", stVersion.nMajor, stVersion.nMinor, stVersion.nRevision);

    ASSERT_OK(MV3D_RGBD_Initialize());

    unsigned int nDevNum = 0;
    ASSERT_OK(MV3D_RGBD_GetDeviceNumber(DeviceType_Ethernet | DeviceType_USB, &nDevNum));
    LOGD("MV3D_RGBD_GetDeviceNumber success! nDevNum:%d.", nDevNum);
    ASSERT(nDevNum);

    // 查找设备
    std::vector<MV3D_RGBD_DEVICE_INFO> devs(nDevNum);
    ASSERT_OK(MV3D_RGBD_GetDeviceList(DeviceType_Ethernet | DeviceType_USB, &devs[0], nDevNum, &nDevNum));
    for (unsigned int i = 0; i < nDevNum; i++)
    {
        LOG("Index[%d]. SerialNum[%s] IP[%s] name[%s].\r\n", i, devs[i].chSerialNumber, devs[i].SpecialInfo.stNetInfo.chCurrentIp, devs[i].chModelName);
    }

    //打开设备
    void* handle = NULL;
    unsigned int nIndex = 0;
    ASSERT_OK(MV3D_RGBD_OpenDevice(&handle, &devs[nIndex]));
    LOGD("OpenDevice success.");

    // 开始工作流程
    ASSERT_OK(MV3D_RGBD_Start(handle));
    LOGD("Start work success.");

    BOOL bExit_Main = FALSE;
    RenderImgWnd depthViewer(1280, 720, "depth");
    MV3D_RGBD_FRAME_DATA stFrameData = { 0 };
    while (!bExit_Main && depthViewer)
    {
        // 获取图像数据
        int nRet = MV3D_RGBD_FetchFrame(handle, &stFrameData, 5000);
        if (MV3D_RGBD_OK == nRet)
        {
            LOGD("MV3D_RGBD_FetchFrame success.");
            RIFrameInfo depth = { 0 };
            RIFrameInfo rgb = { 0 };
            parseFrame(&stFrameData, &depth, &rgb);
            LOGD("rgb: nFrameNum(%d), nheight(%d), nwidth(%d)。", rgb.nFrameNum, rgb.nHeight, rgb.nWidth);
            Mat depth_frame(depth.nHeight, depth.nWidth, CV_16UC1, depth.pData);        //将深度图转化为Mat格式
            imshow("depth_frame", depth_frame);
            Mat rgb_frame(rgb.nHeight, rgb.nWidth, CV_8UC3, rgb.pData);                 //将彩色图转化为Mat格式
            depth_all = depth_frame;
            depthViewer.RenderImage(depth);

            //B、R通道交换，显示正常彩色图像
            Mat A;
            cvtColor(rgb_frame, A, COLOR_BGR2RGB);


           vector<string> class_list;
           ifstream ifs("coco.names");
           string line;

           while (getline(ifs, line))
           {
               class_list.push_back(line);
           }

           // 加载图像。

           Mat frame;
           frame = A;


           // 加载模型。
           Net net;
           net = readNet("models/yolov5m.onnx");

           vector<Mat> detections;
           detections = pre_process(frame, net);
           Mat Z = frame.clone();
           Mat img = post_process(Z, detections, class_list);

           // 放效率信息。
           // 函数 getPerfProfile 返回推理的总时间和每个层的时间
           vector<double> layersTimes;
           double freq = getTickFrequency() / 1000;
           double t = net.getPerfProfile(layersTimes) / freq;
           string label = format("Inference time : %.2f ms", t);
           putText(img, label, Point(20, 40), FONT_FACE, FONT_SCALE, RED);

           imshow("Output", img);
           

        }

        //按任意键退出
        if (_kbhit())
        {
            bExit_Main = TRUE;
        }
    }

    ASSERT_OK(MV3D_RGBD_Stop(handle));
    ASSERT_OK(MV3D_RGBD_CloseDevice(&handle));
    ASSERT_OK(MV3D_RGBD_Release());

    LOGD("Main done!");
    return  0;
}
    
