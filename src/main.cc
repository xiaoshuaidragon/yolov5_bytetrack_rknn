#include <stdio.h>
#include <memory>
#include <sys/time.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "rkYolov5s.hpp"
#include "rknnPool.hpp"
#include "BYTETracker.h"
#include "STrack.h"




const std::vector<std::string> names = {"person", "head_helmet","head","reflective_clothes","smoking","calling","falling","face_mask","car","bicycle","motorcycle",
    "fumes","fire","head_hat","normal_clothes","face","play_phone","other","knife"};

const std::vector<cv::Scalar> colors = {
cv::Scalar(255, 0, 0),    // 蓝色
cv::Scalar(0, 255, 0),    // 绿色
cv::Scalar(0, 0, 255),    // 红色
cv::Scalar(255, 255, 0),  // 青色
cv::Scalar(255, 0, 255),  // 品红色
cv::Scalar(0, 255, 255),  // 黄色
cv::Scalar(192, 192, 192), // 浅灰色
cv::Scalar(128, 0, 0),    // 深红色
cv::Scalar(128, 128, 0),  // 橄榄色
cv::Scalar(0, 128, 0),    // 深绿色
cv::Scalar(128, 0, 128),  // 紫色
cv::Scalar(0, 128, 128),  // 深青色
cv::Scalar(0, 0, 128),    // 深蓝色
cv::Scalar(255, 128, 0),  // 橙色
cv::Scalar(255, 0, 128),  // 红紫色
cv::Scalar(128, 255, 0),  // 黄绿色
cv::Scalar(0, 255, 128),  // 青绿色
cv::Scalar(128, 0, 255),  // 蓝紫色
cv::Scalar(255, 255, 255) // 白色
};

const float person_thres = 0.1;      //0
const float head_helmet_thres = 0.15;        //1
const float head_thres = 0.15;                //2
const float reflective_clothes_thres = 0.5;   //3
const float smoking_thres = 0.15;    //4
const float calling_thres = 0.15;     //5
const float falling_thres = 0.5;       //6
const float face_mask_thres = 0.5;     //7
const float car_thres = 0.5;           //8
const float bicycle_thres = 0.5;        //9
const float motorcycle_thres = 0.5;     //10
const float fumes_thres = 0.5;          //11
const float fire_thres = 0.5;          //12
const float head_hat_thres = 0.15;      //13
const float normal_clothes_thres = 0.5; //14
const float face_thres = 0.5;           //15
const float play_phone_thres = 0.15;    //16
const float other_thres = 0.5;         //17
const float knife_thres = 0.15;         //18




int main(int argc, char **argv)
{
    char *model_name = NULL;
    if (argc != 3)
    {
        printf("Usage: %s <rknn model> <jpg> \n", argv[0]);
        return -1;
    }
    // 参数二，模型所在路径/The path where the model is located
    model_name = (char *)argv[1];
    // 参数三, 视频/摄像头
    char *vedio_name = argv[2];

    // 初始化rknn线程池/Initialize the rknn thread pool
    int threadNum = 3;
    rknnPool<rkYolov5s, cv::Mat, cv::Mat> testPool(model_name, threadNum);
    if (testPool.init() != 0)
    {
        printf("rknnPool init fail!\n");
        return -1;
    }

    cv::namedWindow("Camera FPS");
    cv::VideoCapture capture;
    if (strlen(vedio_name) == 1)
        capture.open((int)(vedio_name[0] - '0'));
    else
        capture.open(vedio_name);

    struct timeval time;
    gettimeofday(&time, nullptr);
    auto startTime = time.tv_sec * 1000 + time.tv_usec / 1000;

    auto total_start = std::chrono::high_resolution_clock::now();

    int frames = 0;
    BYTETracker tracker(50, 50);
    while (capture.isOpened())
    {
        cv::Mat img;
        int video_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
        int video_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
        int video_fps = capture.get(cv::CAP_PROP_FPS);
        int video_frames = capture.get(cv::CAP_PROP_FRAME_COUNT);
        
        if (capture.read(img) == false)
            break;
        auto detect_start = std::chrono::high_resolution_clock::now();
        if (testPool.put(img) != 0)
            break;

        if (frames >= threadNum && testPool.get(img) != 0)
            break;

        


        std::vector<rkYolov5s::DetectionResult> detections;
        std::vector<Object> objects;
        if (testPool.getDetections(detections) == 0) {
            auto detect_end = std::chrono::high_resolution_clock::now();
        float detect_time = std::chrono::duration<float, std::milli>(detect_end - detect_start).count();
        printf("Frame %d - Detect Time: %.2f ms\n", frames, detect_time);
            for (const auto &det : detections) {
                if (det.class_id == 0) {
                    Object obj;
                    obj.label = det.class_id;  // 类别ID
                    obj.prob = det.confidence; // 置信度

                    obj.rect = cv::Rect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);
        
                    objects.push_back(obj);  // 添加到 objects 列表
                }
            }
        }
        auto track_start = std::chrono::high_resolution_clock::now();
        vector<STrack> output_stracks = tracker.update(objects);
        auto track_end = std::chrono::high_resolution_clock::now();
        float track_time = std::chrono::duration<float, std::milli>(track_end - track_start).count();
        printf("Frame %d - Track Time: %.2f ms\n", frames, track_time);
        for(int i = 0; i < output_stracks.size(); i++){
            vector<float> tlwh = output_stracks[i].tlwh;
            std::string conf_str = std::to_string(output_stracks[i].score);
            conf_str = conf_str.substr(0, conf_str.find('.') + 3); 
            std::string label = names[output_stracks[i].label] + ": " + conf_str;
            cv::putText(img, label, cv::Point(tlwh[0], tlwh[1] - 5),cv::FONT_HERSHEY_SIMPLEX, 0.4, colors[output_stracks[i].label], 1);
            cv::putText(img, "id: " + format("%d", output_stracks[i].track_id), cv::Point(tlwh[0], tlwh[1] - 15),cv::FONT_HERSHEY_SIMPLEX, 0.4, colors[output_stracks[i].label], 1);
            cv::rectangle(img,Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), colors[output_stracks[i].label], 2);                
            
        }
        // Show result
        cv::imshow("detect", img);
        if (cv::waitKey(1) == 'q')
            break;
        frames++;
    }

    // 清空rknn线程池/Clear the thread pool
    while (true)
    {
        cv::Mat img;
        if (testPool.get(img) != 0)
            break;
        cv::imshow("Camera FPS", img);
        if (cv::waitKey(1) == 'q') // 延时1毫秒,按q键退出/Press q to exit
            break;
        frames++;
    }

    // gettimeofday(&time, nullptr);
    auto total_end = std::chrono::high_resolution_clock::now();

    // 计算总耗时（单位为秒）
    double total_time = std::chrono::duration<double>(total_end - total_start).count();

    // 计算平均帧率
    double avg_fps = frames / total_time;

    // 输出平均帧率
    printf("Total Frames: %d, Total Time: %.2f seconds, Average FPS: %.2f\n", frames, total_time, avg_fps);
    return 0;
}