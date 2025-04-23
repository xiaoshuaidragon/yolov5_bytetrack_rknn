#ifndef RKYOLOV5S_H
#define RKYOLOV5S_H

#include "rknn_api.h"

#include "opencv2/core/core.hpp"

static void dump_tensor_attr(rknn_tensor_attr *attr);
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
static unsigned char *load_model(const char *filename, int *model_size);
static int saveFloat(const char *file_name, float *output, int element_size);





class rkYolov5s
{
public:
    struct DetectionResult {
        int class_id;       // 类别ID
        std::string name;   // 类别名称
        float confidence;   // 置信度
        int x1, y1, x2, y2; // 检测框坐标 (left, top, right, bottom)
    };

private:
    int ret;
    std::mutex mtx;
    std::string model_path;
    unsigned char *model_data;

    rknn_context ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;
    rknn_input inputs[1];

    int channel, width, height;
    int img_width, img_height;

    float nms_threshold, box_conf_threshold;
    std::vector<DetectionResult> last_detections; // 存储最后一帧的检测结果
public:

    rkYolov5s(const std::string &model_path);
    int init(rknn_context *ctx_in, bool isChild);
    rknn_context *get_pctx();
    cv::Mat infer(cv::Mat &ori_img);
    std::vector<DetectionResult> getLastDetections();
    ~rkYolov5s();

};

#endif