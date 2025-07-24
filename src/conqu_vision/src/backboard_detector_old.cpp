#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <openvino/openvino.hpp>
#include <vector>
#include <algorithm>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

// 全局变量 - 替代成员变量
ros::NodeHandle* g_nh_ptr = nullptr;
image_transport::ImageTransport* g_it_ptr = nullptr;

// 消息过滤器
message_filters::Subscriber<sensor_msgs::Image>* g_rgb_sub_ptr = nullptr;
message_filters::Subscriber<sensor_msgs::Image>* g_depth_sub_ptr = nullptr;
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
typedef message_filters::Synchronizer<SyncPolicy> Synchronizer;
boost::shared_ptr<Synchronizer> g_sync_ptr;

// 发布者
image_transport::Publisher g_image_pub;
image_transport::Publisher g_cropped_depth_pub;

// 深度图信息
int g_depth_width = 0;
int g_depth_height = 0;

// OpenVINO相关
ov::Core g_core;
ov::CompiledModel g_compiled_model;
ov::InferRequest g_infer_request;

// 模型参数
std::string g_model_path;
int g_input_width;
int g_input_height;
float g_confidence_threshold;
std::vector<std::string> g_class_names;

// 图像处理参数
int g_canny_threshold1 = 100;
int g_canny_threshold2 = 200;
double g_epsilon_factor = 0.02;
double g_min_contour_area = 100.0;

// 多帧稳定性
std::vector<cv::Point> g_last_good_approx;
int g_stable_frames = 0;
ros::Time g_last_process_time;

// 霍夫线检测和评分参数
int g_hough_threshold = 30;        // 霍夫线检测阈值
double g_sL_dLU_weight = 15.0;     // 左线dLU权重
double g_sL_dLD_weight = 8.0;      // 左线dLD权重
double g_sL_angle_weight = 40.0;   // 左线angle权重
double g_sR_dRU_weight = 15.0;     // 右线dRU权重
double g_sR_dRD_weight = 8.0;      // 右线dRD权重
double g_sR_angle_weight = 40.0;   // 右线angle权重

// 检测结构体定义
struct Detection {
    cv::Rect bbox;
    float confidence;
    int class_id;
};

// 结构体定义用于存储检测到的重合点
struct ClosePoints {
    cv::Point LU; // 左上角
    cv::Point RU; // 右上角
    cv::Point LD; // 左下角
    cv::Point RD; // 右下角
    int successFlag = 0;
};

// 函数声明
void initOpenVINO();
cv::Mat preprocessImage(const cv::Mat& image);
std::vector<Detection> postprocessDetections(const ov::Tensor& output_tensor, float original_width, float original_height);
cv::Mat drawDetections(const cv::Mat& image, const std::vector<Detection>& detections);
cv::Rect mapRgbRectToDepth(const cv::Rect& rgb_rect, int rgb_width, int rgb_height);
ClosePoints findClosePoints(const std::vector<cv::Vec4i>& lines, int width, int height, int threshold = 10);
void processDetectionsWithDepth(const std::vector<Detection>& detections, int rgb_width, int rgb_height, const cv::Mat& depth_image, const ros::Time& timestamp);
void imageCbWithDepth(const sensor_msgs::ImageConstPtr& rgb_msg, const sensor_msgs::ImageConstPtr& depth_msg);

// 全局变量
pcl::PointCloud<pcl::PointXYZ>::Ptr g_point_cloud;
ros::Subscriber g_pointcloud_sub;

// 点云回调函数
void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    pcl::fromROSMsg(*msg, *g_point_cloud);
}

// 初始化OpenVINO
void initOpenVINO() {
    try {
        ROS_INFO("Loading OpenVINO model: %s", g_model_path.c_str());
        
        // 读取模型
        auto model = g_core.read_model(g_model_path);
        
        // 检查输入输出形状
        auto input_port = model->input();
        auto output_port = model->output();
        
        ROS_INFO("Model input shape: [%ld, %ld, %ld, %ld]", 
            input_port.get_shape()[0], input_port.get_shape()[1],
            input_port.get_shape()[2], input_port.get_shape()[3]);
        
        // 设置GPU设备
        ov::AnyMap config;
        config["PERFORMANCE_HINT"] = "THROUGHPUT";
        config["GPU_ENABLE_LOOP_UNROLLING"] = "YES";
        
        // 编译模型 - 使用Intel GPU
        try {
            g_compiled_model = g_core.compile_model(model, "GPU", config);
            ROS_INFO("Using Intel GPU for inference");
        } catch (const std::exception& e) {
            ROS_WARN("Failed to use GPU, falling back to CPU: %s", e.what());
            g_compiled_model = g_core.compile_model(model, "CPU");
        }
        
        // 创建推理请求
        g_infer_request = g_compiled_model.create_infer_request();
        
    } catch (const std::exception& e) {
        ROS_ERROR("Failed to initialize OpenVINO: %s", e.what());
        ros::shutdown();
    }
}

// 图像预处理
cv::Mat preprocessImage(const cv::Mat& image) {
    cv::Mat rgb_image, resized_image;
    
    // BGR转RGB
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
    
    // 缩放到模型输入尺寸
    cv::resize(rgb_image, resized_image, cv::Size(g_input_width, g_input_height));
    
    // 转换数据类型并归一化
    cv::Mat float_image;
    resized_image.convertTo(float_image, CV_32F, 1.0/255.0);
    
    return float_image;
}

// 后处理检测结果
std::vector<Detection> postprocessDetections(const ov::Tensor& output_tensor, 
                                           float original_width, float original_height) {
    std::vector<Detection> detections;
    
    // 获取输出数据
    const float* data = output_tensor.data<float>();
    
    auto shape = output_tensor.get_shape();
    size_t num_classes_plus_4 = shape[1];
    size_t num_anchors = shape[2];
    size_t num_classes = num_classes_plus_4 - 4;
    
    float scale_x = original_width / g_input_width;
    float scale_y = original_height / g_input_height;

    Detection best_det;
    float best_conf = -1.0f;
    int best_class_id = -1;
    cv::Rect best_bbox;

    for (size_t i = 0; i < num_anchors; ++i) {
        float x_center = data[i];
        float y_center = data[num_anchors + i];
        float width = data[2 * num_anchors + i];
        float height = data[3 * num_anchors + i];

        float max_confidence = 0.0f;
        int max_class_id = -1;
        for (size_t c = 0; c < num_classes; ++c) {
            float confidence = data[(4 + c) * num_anchors + i];
            if (confidence > max_confidence) {
                max_confidence = confidence;
                max_class_id = c;
            }
        }

        if (max_confidence >= g_confidence_threshold && max_confidence > best_conf) {
            float x1 = (x_center - width / 2) * scale_x;
            float y1 = (y_center - height / 2) * scale_y;
            float x2 = (x_center + width / 2) * scale_x;
            float y2 = (y_center + height / 2) * scale_y;

            x1 = std::max(0.0f, std::min(x1, original_width - 1));
            y1 = std::max(0.0f, std::min(y1, original_height - 1));
            x2 = std::max(0.0f, std::min(x2, original_width - 1));
            y2 = std::max(0.0f, std::min(y2, original_height - 1));

            best_bbox = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
            best_conf = max_confidence;
            best_class_id = max_class_id;
        }
    }

    if (best_conf >= g_confidence_threshold) {
        Detection det;
        det.bbox = best_bbox;
        det.confidence = best_conf;
        det.class_id = best_class_id;
        detections.push_back(det);
    }

    return detections;
}

// 绘制检测结果
cv::Mat drawDetections(const cv::Mat& image, const std::vector<Detection>& detections) {
    cv::Mat result = image.clone();
    
    for (const auto& detection : detections) {
        // 绘制边界框
        cv::rectangle(result, detection.bbox, cv::Scalar(0, 255, 0), 2);
        
        // 准备标签文本
        std::string class_name = detection.class_id < g_class_names.size() ? 
            g_class_names[detection.class_id] : "Unknown";
        std::string label = class_name + ": " + 
            std::to_string(static_cast<int>(detection.confidence * 100)) + "%";
        
        // 计算文本尺寸
        int baseline;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
        
        // 绘制文本背景
        cv::Point text_origin(detection.bbox.x, detection.bbox.y - text_size.height - 5);
        cv::rectangle(result, 
            cv::Point(text_origin.x, text_origin.y - text_size.height - 5),
            cv::Point(text_origin.x + text_size.width, text_origin.y + baseline),
            cv::Scalar(0, 255, 0), -1);
        
        // 绘制文本
        cv::putText(result, label, text_origin, cv::FONT_HERSHEY_SIMPLEX, 0.6, 
            cv::Scalar(0, 0, 0), 2);
    }
    
    return result;
}

// 获取指定像素点的3D坐标
cv::Point3f getWorldCoordinates(int pixel_x, int pixel_y) {
    if (!g_point_cloud || g_point_cloud->empty()) {
        ROS_WARN("Point cloud is empty");
        return cv::Point3f(0, 0, 0);
    }
    
    // 检查坐标是否在有效范围内
    if (pixel_x >= 0 && pixel_x < g_point_cloud->width && 
        pixel_y >= 0 && pixel_y < g_point_cloud->height) {
        
        // 获取点云中对应像素的点
        pcl::PointXYZ point = g_point_cloud->at(pixel_x, pixel_y);
        
        // 检查点是否有效（有些点可能是NaN）
        if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) {
            return cv::Point3f(point.x, point.y, point.z);
        }
    }
    
    return cv::Point3f(0, 0, 0); // 无效点返回原点
}

// 将RGB检测框映射到深度图
cv::Rect mapRgbRectToDepth(const cv::Rect& rgb_rect, int rgb_width, int rgb_height) {
    if (g_depth_width == 0 || g_depth_height == 0) {
        ROS_WARN("Depth image dimensions not initialized!");
        return rgb_rect;  // 如果还没有深度图信息，直接返回原始框
    }
    
    // 计算比例
    float scale_x = static_cast<float>(g_depth_width) / static_cast<float>(rgb_width);
    float scale_y = static_cast<float>(g_depth_height) / static_cast<float>(rgb_height);
    
    // 映射矩形
    int width = static_cast<int>(rgb_rect.width * scale_x * 1.2);
    int height = static_cast<int>(rgb_rect.height * scale_y * 1.2);
    int x = static_cast<int>(rgb_rect.x * scale_x - width * 0.1);  // 向左扩展10%
    int y = static_cast<int>(rgb_rect.y * scale_y - height * 0.1);  // 向上扩展10%
    
    // 确保坐标在深度图范围内
    x = std::max(0, std::min(x, g_depth_width - 1));
    y = std::max(0, std::min(y, g_depth_height - 1));
    width = std::min(width, g_depth_width - x);
    height = std::min(height, g_depth_height - y);
    
    return cv::Rect(x, y, width, height);
}

// 检测重合点函数 - 基于test2.py的逻辑
ClosePoints findClosePoints(const std::vector<cv::Vec4i>& lines, int width, int height, int threshold, cv::Mat roi) {
    ClosePoints result_points;
    // 初始化结果点为图像边界外的点
    int depth_LU = 255, depth_RU = 255, depth_LD = 255, depth_RD = 255;

    result_points.LU = cv::Point(width, height);   // 左上角
    result_points.RU = cv::Point(0, height);       // 右上角  
    result_points.LD = cv::Point(width, 0);        // 左下角
    result_points.RD = cv::Point(0, 0);            // 右下角
    
    if (lines.size() < 2) {
        return result_points;
    }
    
    // 计算两点间距离的函数
    auto pointDistance = [](const cv::Point& p1, const cv::Point& p2) -> double {
        return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
    };
    
    // 获取直线端点的函数
    auto getLineEndpoints = [](const cv::Vec4i& line) -> std::vector<cv::Point> {
        return {cv::Point(line[0], line[1]), cv::Point(line[2], line[3])};
    };
    
    size_t n_lines = lines.size();
    
    // 遍历所有直线对
    for (size_t i = 0; i < n_lines; ++i) {
        for (size_t j = i + 1; j < n_lines; ++j) {
            const cv::Vec4i& line1 = lines[i];
            const cv::Vec4i& line2 = lines[j];
            
            int x1 = line1[0], y1 = line1[1], x2 = line1[2], y2 = line1[3];
            int x3 = line2[0], y3 = line2[1], x4 = line2[2], y4 = line2[3];
            
            // 计算斜率
            double k1 = (x2 - x1) != 0 ? static_cast<double>(y2 - y1) / (x2 - x1) : std::numeric_limits<double>::infinity();
            double k2 = (x4 - x3) != 0 ? static_cast<double>(y4 - y3) / (x4 - x3) : std::numeric_limits<double>::infinity();
            
            // 计算夹角的正切值
            double tan_theta;
            if (std::isinf(k1) && std::isinf(k2)) {
                tan_theta = 0; // 两条线都是垂直的
            } else if (std::isinf(k1) || std::isinf(k2)) {
                tan_theta = std::numeric_limits<double>::infinity();
            } else {
                double denominator = 1 + k1 * k2;
                if (std::abs(denominator) < 1e-10) {
                    tan_theta = std::numeric_limits<double>::infinity();
                } else {
                    tan_theta = std::abs((k1 - k2) / denominator);
                }
            }

            // 检查夹角是否在(60°, 120°)范围内
            if (tan_theta > 1.732) { // tan(60°) = √3 ≈ 1.732
                // 获取两条直线的端点
                std::vector<cv::Point> points1 = getLineEndpoints(line1);
                std::vector<cv::Point> points2 = getLineEndpoints(line2);
                
                // 检查是否存在距离小于threshold的端点对
                for (const cv::Point& p1 : points1) {
                    for (const cv::Point& p2 : points2) {
                        double distance = pointDistance(p1, p2);
                        if (distance < threshold) {
                            // 计算平均点
                            cv::Point avg_point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);

                            // 在avg_point周围5x5范围内找到像素值最高的点
                            cv::Point max_point = avg_point;
                            int max_value = -1;

                            // 定义5x5搜索区域
                            int search_radius = 2;
                            int x_start = std::max(0, avg_point.x - search_radius);
                            int y_start = std::max(0, avg_point.y - search_radius);
                            int x_end = std::min(roi.cols, avg_point.x + search_radius + 1);
                            int y_end = std::min(roi.rows, avg_point.y + search_radius + 1);

                            // 提取5x5区域的ROI
                            cv::Rect search_rect(x_start, y_start, x_end - x_start, y_end - y_start);
                            cv::Mat search_roi = roi(search_rect);

                            // 使用OpenCV的minMaxLoc函数找到最大值
                            double minVal, maxVal;
                            cv::Point minLoc, maxLoc_relative;
                            cv::minMaxLoc(search_roi, &minVal, &maxVal, &minLoc, &maxLoc_relative);

                            // 转换回原始坐标系
                            max_point = cv::Point(maxLoc_relative.x + x_start, maxLoc_relative.y + y_start);
                            max_value = static_cast<int>(maxVal);

                            // 使用找到的最高值点计算深度
                            int avg_depth = -max_value;
                            ROS_INFO_THROTTLE(1.0, "Found close points: avg(%d, %d) -> max(%d, %d) with value %d, depth %d", 
                                            avg_point.x, avg_point.y, max_point.x, max_point.y, max_value, avg_depth);
                            
                            avg_point = max_point; // 使用最高值点作为平均点
                            // int avg_depth = 0;
                            double depth_k = 3.0;

                            // 根据位置更新相应的角点
                            double dist_pow = 1; // 可以根据需要设置为参数

                            double LU_score = abs(std::pow(avg_point.x - 0, dist_pow)) + abs(std::pow(avg_point.y - 0, dist_pow)) + avg_depth * depth_k;
                            double LD_score = abs(std::pow(avg_point.x - 0, dist_pow)) + abs(std::pow(avg_point.y - height, dist_pow)) + avg_depth * depth_k;
                            double RU_score = abs(std::pow(avg_point.x - width, dist_pow)) + abs(std::pow(avg_point.y - 0, dist_pow)) + avg_depth * depth_k;
                            double RD_score = abs(std::pow(avg_point.x - width, dist_pow)) + abs(std::pow(avg_point.y - height, dist_pow)) + avg_depth * depth_k;

                            if (avg_point.x < width / 2) { // 左侧
                                if (avg_point.y < height / 2) { // 上侧 - 左上角
                                    if (LU_score < std::pow(result_points.LU.x - 0, dist_pow) + std::pow(result_points.LU.y - 0, dist_pow) + depth_LU * depth_k) {
                                        result_points.LU = avg_point;
                                        depth_LU = avg_depth;
                                        result_points.successFlag |= 0b0001;
                                    }
                                } else { // 下侧 - 左下角
                                    if (LD_score < std::pow(result_points.LD.x - 0, dist_pow) + std::pow(result_points.LD.y - height, dist_pow) + depth_LD * depth_k) {
                                        result_points.LD = avg_point;
                                        depth_LD = avg_depth;
                                        result_points.successFlag |= 0b0010;
                                    }
                                }
                            } else { // 右侧
                                if (avg_point.y < height / 2) { // 上侧 - 右上角
                                    if (RU_score < std::pow(result_points.RU.x - width, dist_pow) + std::pow(result_points.RU.y - 0, dist_pow) + depth_RU * depth_k) {
                                        result_points.RU = avg_point;
                                        depth_RU = avg_depth;
                                        result_points.successFlag |= 0b0100;
                                    }
                                } else { // 下侧 - 右下角
                                    if (RD_score < std::pow(result_points.RD.x - width, dist_pow) + std::pow(result_points.RD.y - height, dist_pow) + depth_RD * depth_k) {
                                        result_points.RD = avg_point;
                                        depth_RD = avg_depth;
                                        result_points.successFlag |= 0b1000;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return result_points;
}

// 处理带有深度信息的检测结果 - 基于test2.py的逻辑
void processDetectionsWithDepth(const std::vector<Detection>& detections, 
                          int rgb_width, int rgb_height,
                          const cv::Mat& depth_image,
                          const ros::Time& timestamp) {
    ROS_INFO("Processing detections with depth image");
    ROS_INFO("Input depth image: %dx%d, type: %d, channels: %d", 
             depth_image.cols, depth_image.rows, depth_image.type(), depth_image.channels());
    
    // 检查深度图的统计信息
    double minVal, maxVal;
    cv::minMaxLoc(depth_image, &minVal, &maxVal);
    ROS_INFO("Depth image stats - Min: %.2f, Max: %.2f", minVal, maxVal);
    
    // 添加帧率控制，最多10Hz处理频率
    if ((ros::Time::now() - g_last_process_time).toSec() < 0.1) {
        return;
    }
    g_last_process_time = ros::Time::now();
    
    for (const auto& detection : detections) {
        // 映射检测框到深度图
        cv::Rect depth_rect = mapRgbRectToDepth(detection.bbox, rgb_width, rgb_height);
        
        // 提取该区域的深度图
        cv::Mat roi = depth_image(depth_rect).clone();
        
        // 先将16位深度图转换为8位可视化图像
        cv::Mat roi_8bit;
        roi.convertTo(roi_8bit, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
        
        // 参照test2.py: 设置0值为255，然后反转
        cv::Mat roi_processed;
        roi_8bit.copyTo(roi_processed);
        roi_processed.setTo(255, roi_processed == 0);
        roi_processed = 255 - roi_processed;
        
        // 获取图像尺寸
        int height = roi.rows;
        int width = roi.cols;
        
        // 转换为8位灰度图
        cv::Mat roi_gray;
        if (roi_processed.channels() == 3) {
            cv::cvtColor(roi_processed, roi_gray, cv::COLOR_BGR2GRAY);
        } else {
            roi_processed.convertTo(roi_gray, CV_8U);
        }
        
        // 保存原始图像用于显示
        cv::Mat img_origin;
        cv::cvtColor(roi_gray, img_origin, cv::COLOR_GRAY2BGR);
        
        // 参照test2.py: 直方图均衡化增强对比度
        cv::equalizeHist(roi_gray, roi_gray);
        
        // 转换为BGR用于后续处理
        cv::Mat img;
        cv::cvtColor(roi_gray, img, cv::COLOR_GRAY2BGR);
        
        // 参照test2.py: GrabCut前景分割
        cv::Mat mask = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
        cv::Mat bgdModel = cv::Mat::zeros(1, 65, CV_64FC1);
        cv::Mat fgdModel = cv::Mat::zeros(1, 65, CV_64FC1);
        cv::Rect rect(20, 20, img.cols - 40, img.rows - 40);
        
        if (rect.width > 0 && rect.height > 0) {
            cv::grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv::GC_INIT_WITH_RECT);
            
            // 创建mask2
            cv::Mat mask2;
            cv::compare(mask, cv::GC_FGD, mask2, cv::CMP_EQ);
            mask2.convertTo(mask2, CV_8U);
            mask2 = 255 - mask2; // 反转mask
            mask2.convertTo(mask2, CV_8U, 1.0/255.0);
            
            // 应用mask
            for (int i = 0; i < img.rows; ++i) {
                for (int j = 0; j < img.cols; ++j) {
                    cv::Vec3b& pixel = img.at<cv::Vec3b>(i, j);
                    uchar mask_val = mask2.at<uchar>(i, j);
                    pixel[0] *= mask_val;
                    pixel[1] *= mask_val;
                    pixel[2] *= mask_val;
                }
            }
        }
        
        // 边缘检测预处理
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        
        // 参照test2.py: 自适应高斯模糊，核大小基于图像高度
        int ksize = static_cast<int>(height * 0.06);
        if (ksize % 2 == 0) ksize += 1;
        if (ksize < 3) ksize = 3;
        cv::GaussianBlur(gray, gray, cv::Size(ksize, ksize), 0);
        
        // 参照test2.py: Canny边缘检测
        cv::Mat canny;
        cv::Canny(gray, canny, 80, 100);
        
        // 参照test2.py: 膨胀操作
        ksize = static_cast<int>(height * 0.02);
        if (ksize % 2 == 0) ksize += 1;
        if (ksize < 3) ksize = 3;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ksize, ksize));
        cv::dilate(canny, canny, kernel);
        
        // 参照test2.py: 霍夫线检测，参数基于图像宽度
        std::vector<cv::Vec4i> lines;
        int hough_threshold = 100;
        int min_line_length = static_cast<int>(width * 0.25);
        int max_line_gap = static_cast<int>(width * 0.5);
        cv::HoughLinesP(canny, lines, 2, CV_PI / 90, hough_threshold, min_line_length, max_line_gap);
        
        // 找出重合的端点，阈值基于图像高度
        int close_threshold = static_cast<int>(height * 0.1);
        ClosePoints close_points = findClosePoints(lines, width, height, close_threshold, roi_gray);
        if (close_points.successFlag != 0b1111) return;
        
        // 在原始图像上标记重合点（参照test2.py）
        std::vector<cv::Point> corner_points = {close_points.LU, close_points.RU, close_points.LD, close_points.RD};
        for (const cv::Point& point : corner_points) {
            // 检查点是否在有效范围内（不是初始化的边界值）
            if (point.x < width && point.y < height && point.x >= 0 && point.y >= 0) {
                cv::circle(img_origin, point, 4, cv::Scalar(0, 0, 255), -1);  // 红色实心圆
                cv::circle(img_origin, point, 2, cv::Scalar(255, 255, 255), 2);  // 白色边框
            }
        }
        
        // 输出检测到的角点信息
        ROS_INFO_THROTTLE(1.0, "Corner points: LU(%d,%d), RU(%d,%d), LD(%d,%d), RD(%d,%d)",
                         close_points.LU.x, close_points.LU.y,
                         close_points.RU.x, close_points.RU.y,
                         close_points.LD.x, close_points.LD.y,
                         close_points.RD.x, close_points.RD.y);
        
        // 发布处理后的图像（显示检测线条和重合点）
        cv_bridge::CvImage cropped_depth_msg;
        cropped_depth_msg.header.stamp = timestamp;
        cropped_depth_msg.encoding = sensor_msgs::image_encodings::BGR8;
        cropped_depth_msg.image = img_origin; // 显示带有角点标记的原始图像
        g_cropped_depth_pub.publish(cropped_depth_msg.toImageMsg());
    }
}

// 找到深度阈值，使得深度大于该阈值的像素占10%
uint16_t findDepthThresholdForPercentage(const cv::Mat& depth_image, double percentage = 0.1) {
    // 计算直方图
    int hist_size = 8000; // 16位深度图的最大值
    float range[] = {0, 8000};
    const float* hist_range = {range};
    
    cv::Mat hist;
    cv::calcHist(&depth_image, 1, 0, cv::Mat(), hist, 1, &hist_size, &hist_range);
    
    // 计算总像素数（排除0值）
    int total_pixels = depth_image.rows * depth_image.cols;
    int valid_pixels = total_pixels - static_cast<int>(hist.at<float>(0)); // 排除深度为0的像素
    
    // 计算目标像素数
    int target_pixels = static_cast<int>(valid_pixels * percentage);
    
    // 从高深度值开始累加，找到阈值
    int cumulative_pixels = 0;
    for (int i = hist_size - 1; i >= 0; i -= 10) {
        cumulative_pixels += static_cast<int>(hist.at<float>(i));
        if (cumulative_pixels >= target_pixels) {
            ROS_INFO("找到深度阈值: %d，对应像素比例: %.2f%%", 
                    i, (double)cumulative_pixels / valid_pixels * 100.0);
            return static_cast<uint16_t>(i);
        }
    }
    
    return 0; // 如果没找到合适阈值
}

// 处理同步的RGB和深度图像
void imageCbWithDepth(const sensor_msgs::ImageConstPtr& rgb_msg, const sensor_msgs::ImageConstPtr& depth_msg) {
    cv_bridge::CvImagePtr cv_rgb_ptr, cv_depth_ptr;
    try {
        cv_rgb_ptr = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8);
        cv_depth_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    
    cv::Mat rgb_image = cv_rgb_ptr->image;
    cv::Mat depth_image = cv_depth_ptr->image;

    cv::imwrite("/home/rc1/cqc/R2_Real_ws/rgb_image.png", rgb_image);
    cv::imwrite("/home/rc1/cqc/R2_Real_ws/depth_image.png", depth_image);

    // 调整depth图尺寸与rgb一致
    cv::Mat depth_resized;
    cv::resize(depth_image, depth_resized, rgb_image.size());

    // 找到10%像素对应的深度阈值
    // uint16_t depth_threshold = findDepthThresholdForPercentage(depth_resized, 0.1);

    // depth_resized.setTo(8000, depth_resized == 0);

    for (int y = 0; y < rgb_image.rows; ++y) {
        for (int x = 0; x < rgb_image.cols; ++x) {
            uint16_t depth_val = depth_resized.at<uint16_t>(y, x);
            cv::Vec3b& pixel = rgb_image.at<cv::Vec3b>(y, x);

            if (depth_val == 0) {
                pixel = 0;
                continue;
            }

            // double scale = pow(static_cast<double>(8000 - depth_val) / 8000.0, 4.0);
            double scale = 1.0 - static_cast<double>(depth_val) / 8000.0; // 假设深度范围是0到8000
            for (int c = 0; c < 3; ++c) {
                pixel[c] = std::min(static_cast<uchar>(pixel[c] * scale), static_cast<uchar>(255));
            }
        }
    }

    cv::Mat canny_image;
    cv::Canny(rgb_image, canny_image, 100, 200);

    std::vector<cv::Vec4i> lines;
    int hough_threshold = 80;
    int min_line_length = static_cast<int>(canny_image.cols * 0.3);
    int max_line_gap = static_cast<int>(canny_image.cols * 0.1);
    cv::HoughLinesP(canny_image, lines, 1, CV_PI / 90, hough_threshold, min_line_length, max_line_gap);

    for (const auto& line : lines) {
        cv::line(rgb_image, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 255, 0), 2);
    }

    // cv::imwrite("/home/rc1/cqc/R2_Real_ws/rgb_scaled_by_depth.png", rgb_image);

    // double minVal, maxVal;
    // cv::Point minLoc, maxLoc;
    // cv::minMaxLoc(depth_image, &minVal, &maxVal, &minLoc, &maxLoc);
    // ROS_INFO("深度图最大值: %.2f，位置: (%d, %d)", minVal, maxLoc.x, maxLoc.y);

    // cv::Mat depth_normalized;
    // depth_image.convertTo(depth_normalized, CV_8U, 0.031875, 0.0);  // 深度范围: 0 ~ 8000

    // depth_resized = 255 - depth_resized;
    // depth_resized.setTo(0, depth_resized == 255); // 把背景变成纯黑

    // 用深度图的

    // // 构造带透明度的RGBA图像
    // cv::Mat rgba_image;
    // cv::cvtColor(rgb_image, rgba_image, cv::COLOR_BGR2BGRA);

    // // 用depth作为alpha通道
    // for (int y = 0; y < rgba_image.rows; ++y) {
    //     for (int x = 0; x < rgba_image.cols; ++x) {
    //         uchar depth_val = depth_resized.at<uchar>(y, x);
    //         int alpha = std::pow(256.0, static_cast<double>(depth_val) / 200.0) - 1.0;
    //         alpha = std::max(0, std::min(alpha, 255));
    //         rgba_image.at<cv::Vec4b>(y, x)[3] = static_cast<uchar>(alpha);
    //     }
    // }

    // // 可选：保存融合后的图像
    cv::imwrite("/home/rc1/cqc/R2_Real_ws/rgb_image_2.png", rgb_image);
    return;

    // 更新深度图尺寸
    g_depth_width = depth_image.cols;
    g_depth_height = depth_image.rows;
    
    float rgb_width = rgb_image.cols;
    float rgb_height = rgb_image.rows;
    
    try {
        // 预处理图像
        cv::Mat preprocessed = preprocessImage(rgb_image);
        
        // 创建输入张量
        ov::Tensor input_tensor = ov::Tensor(ov::element::f32, {1, 3, static_cast<size_t>(g_input_height), static_cast<size_t>(g_input_width)});
        float* input_data = input_tensor.data<float>();
        
        // 复制数据到张量 (HWC -> CHW)
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < g_input_height; ++h) {
                for (int w = 0; w < g_input_width; ++w) {
                    input_data[c * g_input_height * g_input_width + h * g_input_width + w] = 
                        preprocessed.at<cv::Vec3f>(h, w)[c];
                }
            }
        }
        
        // 设置输入张量
        g_infer_request.set_input_tensor(input_tensor);
        
        // 运行推理
        g_infer_request.infer();
        
        // 获取输出
        auto output_tensor = g_infer_request.get_output_tensor();
        
        // 后处理
        std::vector<Detection> detections = postprocessDetections(output_tensor, 
            rgb_width, rgb_height);
        
        // 映射检测结果到深度图并处理
        processDetectionsWithDepth(detections, rgb_width, rgb_height, depth_image, depth_msg->header.stamp);
        
        // 绘制检测结果
        cv::Mat result_image = drawDetections(rgb_image, detections);
        
        // 发布结果
        cv_bridge::CvImage out_msg;
        out_msg.header = rgb_msg->header;
        out_msg.encoding = sensor_msgs::image_encodings::BGR8;
        out_msg.image = result_image;
        
        g_image_pub.publish(out_msg.toImageMsg());
        
    } catch (const std::exception& e) {
        ROS_ERROR("Inference failed: %s", e.what());
    }
}

// 主函数
int main(int argc, char** argv) {
    ros::init(argc, argv, "backboard_detector");
    
    // 初始化全局变量
    ros::NodeHandle nh;
    g_nh_ptr = &nh;
    
    image_transport::ImageTransport it(nh);
    g_it_ptr = &it;
    
    g_last_process_time = ros::Time::now();
    
    // 读取参数
    ros::param::param<std::string>("~model_path", g_model_path, "/home/rc1/cqc/R2_Real_ws/hoop.onnx");
    ros::param::param<int>("~input_width", g_input_width, 640);
    ros::param::param<int>("~input_height", g_input_height, 640);
    ros::param::param<float>("~confidence_threshold", g_confidence_threshold, 0.7f);
    
    // 初始化类别名称
    g_class_names = {"basketball_hoop"};
    
    // 初始化订阅者
    g_rgb_sub_ptr = new message_filters::Subscriber<sensor_msgs::Image>(nh, "/camera/color/image_raw", 1);
    g_depth_sub_ptr = new message_filters::Subscriber<sensor_msgs::Image>(nh, "/camera/depth/image_raw", 1);
    
    // 使用近似时间同步策略
    g_sync_ptr.reset(new Synchronizer(SyncPolicy(10), *g_rgb_sub_ptr, *g_depth_sub_ptr));
    g_sync_ptr->registerCallback(boost::bind(&imageCbWithDepth, _1, _2));

    // 初始化点云
    g_point_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    
    // 订阅点云话题
    g_pointcloud_sub = nh.subscribe("/camera/depth_registered/points", 1, pointCloudCallback);
    
    // 初始化发布者
    g_image_pub = it.advertise("/detection/result", 1);
    g_cropped_depth_pub = it.advertise("/detection/result_keypoints", 1);
    
    // 初始化OpenVINO
    initOpenVINO();
    
    ROS_INFO("Backboard detector initialized successfully");
    
    ros::spin();
    
    // 清理资源
    delete g_rgb_sub_ptr;
    delete g_depth_sub_ptr;
    
    return 0;
}
