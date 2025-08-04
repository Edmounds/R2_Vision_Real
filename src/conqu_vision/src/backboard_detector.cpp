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
#include <tf/transform_listener.h>
#include <geometry_msgs/PointStamped.h>

// OpenVINO相关
ov::Core core;
ov::CompiledModel compiled_model;
ov::InferRequest infer_request;

std::string model_path;
int input_width;
int input_height;

image_transport::Publisher image_pub, yolo_pub;
ros::Publisher target_pub;  // 目标点发布者

// 初始化OpenVINO
void initOpenVINO() {
    try {
        ROS_INFO("Loading OpenVINO model: %s", model_path.c_str());
        
        // 读取模型
        auto model = core.read_model(model_path);
        
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
            compiled_model = core.compile_model(model, "GPU", config);
            ROS_INFO("Using Intel GPU for inference");
        } catch (const std::exception& e) {
            ROS_WARN("Failed to use GPU, falling back to CPU: %s", e.what());
            compiled_model = core.compile_model(model, "CPU");
        }
        
        // 创建推理请求
        infer_request = compiled_model.create_infer_request();
        
    } catch (const std::exception& e) {
        ROS_ERROR("Failed to initialize OpenVINO: %s", e.what());
        ros::shutdown();
    }
}

// 检测图像
cv::Rect detectImg(cv::Mat img_input){
    /* ==================== 预处理图像 START ==================== */

    cv::Mat rgb_image, resized_image;
    
    // BGR转RGB
    cv::cvtColor(img_input, rgb_image, cv::COLOR_BGR2RGB);
    
    // 缩放到模型输入尺寸
    cv::resize(rgb_image, resized_image, cv::Size(input_width, input_height));
    
    // 转换数据类型并归一化
    cv::Mat preprocessed;
    resized_image.convertTo(preprocessed, CV_32F, 1.0/255.0);

    /* ==================== 预处理图像 END ==================== */


    /* ==================== YOLO识别 START ==================== */

    // 创建输入张量
    ov::Tensor input_tensor = ov::Tensor(ov::element::f32, {1, 3, static_cast<size_t>(input_height), static_cast<size_t>(input_width)});
    float* input_data = input_tensor.data<float>();
    
    // 复制数据到张量 (HWC -> CHW)
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < input_height; ++h) {
            for (int w = 0; w < input_width; ++w) {
                input_data[c * input_height * input_width + h * input_width + w] = 
                    preprocessed.at<cv::Vec3f>(h, w)[c];
            }
        }
    }
    
    // 设置输入张量
    infer_request.set_input_tensor(input_tensor);
    // 运行推理
    infer_request.infer();
    // 获取输出
    auto output_tensor = infer_request.get_output_tensor();

    /* ==================== YOLO识别 END ==================== */


    /* ==================== 后处理识别结果 START ==================== */

    const float* data = output_tensor.data<float>();
    auto shape = output_tensor.get_shape();
    size_t num_classes_plus_4 = shape[1];
    size_t num_anchors = shape[2];
    size_t num_classes = num_classes_plus_4 - 4;

    float scale_x = static_cast<float>(img_input.cols) / input_width;
    float scale_y = static_cast<float>(img_input.rows) / input_height;

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

        // ROS_INFO("111111");

        if (max_confidence > best_conf && max_confidence > 0.1f) {
            float x1 = (x_center - width / 2) * scale_x;
            float y1 = (y_center - height / 2) * scale_y;
            float x2 = (x_center + width / 2) * scale_x;
            float y2 = (y_center + height / 2) * scale_y;

            x1 = std::max(0.0f, std::min(x1, img_input.cols - 1.0f));
            y1 = std::max(0.0f, std::min(y1, img_input.rows - 1.0f));
            x2 = std::max(0.0f, std::min(x2, img_input.cols - 1.0f));
            y2 = std::max(0.0f, std::min(y2, img_input.rows - 1.0f));

            best_bbox = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
            best_conf = max_confidence;
            best_class_id = max_class_id;
        }
    }

    /* ==================== 后处理识别结果 END ==================== */

    return best_bbox;
}

cv::Vec4i mergeLines(const std::vector<cv::Vec4i>& lines) {
    if (lines.empty()) return cv::Vec4i();

    int x1 = lines[0][0], y1 = lines[0][1], x2 = lines[0][2], y2 = lines[0][3];
    for (const auto& line : lines) {
        
    }
    return cv::Vec4i(x1, y1, x2, y2);
}

struct Line {
    double k;
    double b;
    cv::Point2i left;
    cv::Point2i right;
};

// CoreCallBack 回调函数
void ccb(
    const sensor_msgs::ImageConstPtr& rgb_msg, 
    const sensor_msgs::ImageConstPtr& depth_msg, 
    const sensor_msgs::PointCloud2ConstPtr& pcl_msg
){
    // ROS_INFO("Received RGB and Depth images, processing...");

    cv_bridge::CvImagePtr cv_rgb_ptr, cv_depth_ptr;
    cv_rgb_ptr = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8);
    cv_depth_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
    cv::Mat rgb_image = cv_rgb_ptr->image;
    cv::Mat depth_image = cv_depth_ptr->image;

    // 点云数据转换为PCL格式
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*pcl_msg, *cloud);

    // 调整depth图尺寸与rgb一致
    cv::Mat depth_resized;
    cv::resize(depth_image, depth_resized, rgb_image.size());

    cv::Mat fused_image = rgb_image.clone();
    for (int y = 0; y < fused_image.rows; ++y) {
        for (int x = 0; x < fused_image.cols; ++x) {
            uint16_t depth_val = depth_resized.at<uint16_t>(y, x);
            cv::Vec3b& pixel = fused_image.at<cv::Vec3b>(y, x);

            // 把距离过远的像素直接设为黑色
            if (depth_val == 0) {
                pixel = 0;
                continue;
            }

            // double scale = pow(static_cast<double>(8000 - depth_val) / 8000.0, 4.0);
            double scale = 1.0 - static_cast<double>(depth_val) / 8000.0;
            for (int c = 0; c < 3; ++c) {
                pixel[c] = std::min(static_cast<uchar>(pixel[c] * scale), static_cast<uchar>(255));
            }
        }
    }

    // 检测图像
    cv::Rect detected_bbox = detectImg(rgb_image);
    if (detected_bbox.width == 0 || detected_bbox.height == 0) {
        return;
    }

    // 在输入图上框出识别范围并显示
    cv::rectangle(rgb_image, detected_bbox, cv::Scalar(0, 255, 0), 2);
    cv_bridge::CvImage out_msg_yolo;
    out_msg_yolo.header = rgb_msg->header;
    out_msg_yolo.encoding = sensor_msgs::image_encodings::BGR8;
    out_msg_yolo.image = rgb_image;
    yolo_pub.publish(out_msg_yolo.toImageMsg());
    
    // 计算新的宽高（扩大到105%）
    int new_width = static_cast<int>(detected_bbox.width * 1.05);
    int new_height = static_cast<int>(detected_bbox.height * 1.05);

    // 保持中心点不变
    int center_x = detected_bbox.x + detected_bbox.width / 2;
    int center_y = detected_bbox.y + detected_bbox.height / 2;
    int new_x = center_x - new_width / 2;
    int new_y = center_y - new_height / 2;

    // 限制边界
    new_x = std::max(0, new_x);
    new_y = std::max(0, new_y);
    if (new_x + new_width > rgb_image.cols) new_width = rgb_image.cols - new_x;
    if (new_y + new_height > rgb_image.rows) new_height = rgb_image.rows - new_y;

    cv::Rect enlarged_bbox(new_x, new_y, new_width, new_height);

    // 裁剪检测到的区域
    cv::Mat cropped_image = fused_image(enlarged_bbox).clone();

    // 转换到 HSV 颜色空间
    cv::Mat hsv_image;
    cv::cvtColor(cropped_image, hsv_image, cv::COLOR_BGR2HSV);
    // 将红色外像素转换为黑色
    cv::Mat mask;
    // 检测红色区域：H分量>240或<15
    cv::Mat mask1, mask2;
    cv::inRange(hsv_image, cv::Scalar(0, 100, 100), cv::Scalar(15, 255, 255), mask1);
    cv::inRange(hsv_image, cv::Scalar(165, 100, 100), cv::Scalar(180, 255, 255), mask2);
    cv::bitwise_or(mask1, mask2, mask);
    cv::bitwise_and(cropped_image, mask, cropped_image);

    // 进行 Canny 边缘检测
    cv::Mat canny_image;
    cv::Canny(cropped_image, canny_image, 100, 200);

    // 进行霍夫直线拟合
    std::vector<cv::Vec4i> lines;
    int hough_threshold = 50;
    int min_line_length = static_cast<int>(canny_image.cols * 0.4);
    int max_line_gap = static_cast<int>(canny_image.cols * 0.3);
    cv::HoughLinesP(canny_image, lines, 1, CV_PI / 90, hough_threshold, min_line_length, max_line_gap);

    // 合并线段
    std::vector<Line> lines_params;
    for (const auto& line : lines) {
        cv::Point2i left, right;
        if (line[1] < line[3]) {
            left = cv::Point2i(line[0], line[1]);
            right = cv::Point2i(line[2], line[3]);
        } else {
            left = cv::Point2i(line[2], line[3]);
            right = cv::Point2i(line[0], line[1]);
        }

        if (std::min(left.y, right.y) < cropped_image.rows * 0.5) continue; // 忽略过低的线段
        // cv::line(cropped_image, left, right, cv::Scalar(255, 0, 0), 1);

        // 计算斜率和截距
        double k = static_cast<double>(right.y - left.y) / (right.x - left.x);
        double b = left.y - k * left.x;
        // ROS_INFO("Line: k = %.2f, b = %.2f, left = (%d, %d), right = (%d, %d)", 
        //     k, b, left.x, left.y, right.x, right.y);

        // 合并邻近的直线
        bool merged = false;
        for (Line& line_prev : lines_params) {
            if (abs(line_prev.k - k) < 0.05 and abs(line_prev.b - b) < cropped_image.rows * 0.05) {
                if (left.x < line_prev.left.x) line_prev.left = left;
                if (right.x > line_prev.right.x) line_prev.right = right;

                merged = true;
                break;
            }
        }

        if (!merged){
            lines_params.push_back(Line({k, b, cv::Point(line[0], line[1]), cv::Point(line[2], line[3])}));
        }
    }

    // 找出分数最高的线段
    int max_score = std::numeric_limits<int>::min();
    double dist_k = 2.0;
    Line max_line;

    for (const Line line_filtered : lines_params){
        if (
            line_filtered.left.x > cropped_image.cols * 0.3
            or line_filtered.right.x < cropped_image.cols * 0.7
            or std::max(line_filtered.left.y, line_filtered.right.y) < cropped_image.rows * 0.9
        ) continue;

        // 分数计算: 长度 - 中心点与图像下-中边缘点的距离 * 距离权重
        if (cv::norm(line_filtered.left - line_filtered.right) - cv::norm(
            (line_filtered.left + line_filtered.right) / 2 -
            cv::Point(cropped_image.cols / 2, cropped_image.rows)
        ) * dist_k > max_score) {
            max_score = cv::norm(line_filtered.left - line_filtered.right);
            max_line = line_filtered;
        }
    }

    if (max_score == std::numeric_limits<int>::min()) {
        return;
    }

    // ROS_INFO("Max line score: %d", max_score);
    // ROS_INFO("Max line left: (%d, %d), right: (%d, %d)", 
    //     max_line.left.x, max_line.left.y, max_line.right.x, max_line.right.y);
    cv::line(cropped_image, max_line.left, max_line.right, cv::Scalar(0, 255, 0), 2);

    // 计算max_line.left和max_line.right在原始图像中的坐标
    int left_img_x = enlarged_bbox.x + max_line.left.x;
    int left_img_y = enlarged_bbox.y + max_line.left.y;
    int right_img_x = enlarged_bbox.x + max_line.right.x;
    int right_img_y = enlarged_bbox.y + max_line.right.y;

    // 在max_line.left和max_line.right周围半径5个点内寻找深度最小的点
    int search_radius = 5;

    // 检查坐标有效性
    if (!(
        left_img_x >= search_radius && left_img_x < (cloud->width - search_radius) &&
        left_img_y >= search_radius && left_img_y < (cloud->height - search_radius) &&
        right_img_x >= search_radius && right_img_x < (cloud->width - search_radius) &&
        right_img_y >= search_radius && right_img_y < (cloud->height - search_radius)
    )) return;

    // 在左侧点周围搜索深度最小的点
    float min_depth_left = std::numeric_limits<float>::max();
    int best_left_x = left_img_x, best_left_y = left_img_y;
    
    for (int dy = -search_radius; dy <= search_radius; ++dy) {
        for (int dx = -search_radius; dx <= search_radius; ++dx) {
            if (dx*dx + dy*dy <= search_radius*search_radius) { // 圆形搜索范围
                int search_x = left_img_x + dx;
                int search_y = left_img_y + dy;
                int search_index = search_y * cloud->width + search_x;
                
                pcl::PointXYZ search_pt = cloud->points[search_index];
                
                // 检查点是否有效且深度更小
                if (!std::isnan(search_pt.z) && search_pt.z > 0 && search_pt.z < min_depth_left) {
                    min_depth_left = search_pt.z;
                    best_left_x = search_x;
                    best_left_y = search_y;
                }
            }
        }
    }
    
    // 在右侧点周围搜索深度最小的点
    float min_depth_right = std::numeric_limits<float>::max();
    int best_right_x = right_img_x, best_right_y = right_img_y;
    
    for (int dy = -search_radius; dy <= search_radius; ++dy) {
        for (int dx = -search_radius; dx <= search_radius; ++dx) {
            if (dx*dx + dy*dy <= search_radius*search_radius) { // 圆形搜索范围
                int search_x = right_img_x + dx;
                int search_y = right_img_y + dy;
                int search_index = search_y * cloud->width + search_x;
                
                pcl::PointXYZ search_pt = cloud->points[search_index];
                
                // 检查点是否有效且深度更小
                if (!std::isnan(search_pt.z) && search_pt.z > 0 && search_pt.z < min_depth_right) {
                    min_depth_right = search_pt.z;
                    best_right_x = search_x;
                    best_right_y = search_y;
                }
            }
        }
    }
    
    // 获取最终的最佳点坐标
    int best_left_index = best_left_y * cloud->width + best_left_x;
    int best_right_index = best_right_y * cloud->width + best_right_x;
    
    pcl::PointXYZ best_left_pt = cloud->points[best_left_index];
    pcl::PointXYZ best_right_pt = cloud->points[best_right_index];
    pcl::PointXYZ best_mid_pt(
        (best_left_pt.x + best_right_pt.x) / 2.0f,
        (best_left_pt.y + best_right_pt.y) / 2.0f,
        (best_left_pt.z + best_right_pt.z) / 2.0f
    );

    // ROS_INFO("Left: (%.2f, %.2f, %.2f)", best_left_pt.x, best_left_pt.y, best_left_pt.z);
    // ROS_INFO("Right: (%.2f, %.2f, %.2f)", best_right_pt.x, best_right_pt.y, best_right_pt.z);
    // ROS_INFO("Middle: (%.2f, %.2f, %.2f)", best_mid_pt.x, best_mid_pt.y, best_mid_pt.z);

    // 通过 TF 变换，把相机坐标系的 best_mid_pt 坐标变换到底盘坐标系中
    tf::TransformListener listener;
    tf::StampedTransform transform;
    try {
        listener.waitForTransform("/base_link", "/camera_link", ros::Time(0), ros::Duration(3.0));
        listener.lookupTransform("/base_link", "/camera_link", ros::Time(0), transform);
    } catch (tf::TransformException& ex) {
        ROS_WARN("%s", ex.what());
    }

    tf::Vector3 mid_pt_camera(best_mid_pt.x, best_mid_pt.y, best_mid_pt.z);
    tf::Vector3 mid_pt_base = transform * mid_pt_camera;

    ROS_INFO("Base: (%.2f, %.2f, %.2f)", mid_pt_base.x(), mid_pt_base.y(), mid_pt_base.z());

    // 发布目标点坐标
    geometry_msgs::PointStamped target_msg;
    target_msg.header.stamp = ros::Time::now();
    target_msg.header.frame_id = "base_link";
    target_msg.point.x = mid_pt_base.x();
    target_msg.point.y = mid_pt_base.y();
    target_msg.point.z = mid_pt_base.z();
    target_pub.publish(target_msg);

    // 发布结果
    cv_bridge::CvImage out_msg;
    out_msg.header = rgb_msg->header;
    out_msg.encoding = sensor_msgs::image_encodings::BGR8;
    out_msg.image = cropped_image;
    image_pub.publish(out_msg.toImageMsg());
}

// 主函数
int main(int argc, char** argv) {
    ros::init(argc, argv, "backboard_detector");
    ros::NodeHandle nh;
    
    // 读取参数
    ros::param::param<std::string>("~model_path", model_path, "/home/rc1/cqc/R2_Real_ws/hoop.onnx");
    ros::param::param<int>("~input_width", input_width, 640);
    ros::param::param<int>("~input_height", input_height, 640);
    
    // 创建订阅者
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "/camera/color/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "/camera/depth/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::PointCloud2> pcl_sub(nh, "/camera/depth/points", 1);
    
    // 创建三个topic的同步策略
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::PointCloud2> SyncPolicy;
    message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(10), rgb_sub, depth_sub, pcl_sub);
    sync.registerCallback(boost::bind(&ccb, _1, _2, _3));
    
    // 创建发布者
    image_transport::ImageTransport it(nh);
    image_pub = it.advertise("/detection/result", 1);
    yolo_pub = it.advertise("/detection/yolo", 1);
    target_pub = nh.advertise<geometry_msgs::PointStamped>("/target", 1);

    // 初始化OpenVINO
    initOpenVINO();
    
    ROS_INFO("Backboard detector initialized successfully");
    ros::spin();
    
    return 0;
}
