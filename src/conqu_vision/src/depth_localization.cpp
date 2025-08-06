#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <microhttpd.h>
#include <mutex>
#include <string>
#include <thread>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <geometry_msgs/PointStamped.h>

#define PORT 8080
#define IMG_WIDTH 640
#define IMG_HEIGHT 480
#define LINE_X 360
#define DEPTH_BUFFER_SIZE 100

// 全局变量：原始深度值和卡尔曼滤波后的深度值
int depth = 0;
int depth_filtered = 0;  // 滤波后的值

std::vector<int> depth_history;
ros::Publisher target_pub;  // 目标点发布者

// 卡尔曼滤波参数
float kf_gain = 0.0f;        // 卡尔曼增益K
float kf_estimate = 0.0f;    // 当前估计值 x_hat
float kf_error = 1.0f;       // 估计误差协方差P

// 噪声参数（需根据实际数据调整！）
const float Q = 1.5f;       // 过程噪声方差（模型不确定性）
const float R = 8000.0f;        // 观测噪声方差（传感器噪声）

// 卡尔曼滤波更新函数
void kalman_update(float measurement) {
    if (kf_estimate < 50.0f) { // depth_filter_min
        ROS_INFO("%.4f", measurement);
        kf_estimate = measurement;
        return;
    }

    // 1. 预测阶段
    float predict_estimate = kf_estimate;          // x_k^- = A * x_{k-1} (A=1)
    float predict_error = kf_error + Q;            // P_k^- = A^2 * P_{k-1} + Q (A=1)

    // 2. 更新阶段
    kf_gain = predict_error / (predict_error + R); // K = P_k^- / (P_k^- + R)
    kf_estimate = predict_estimate + kf_gain * (measurement - predict_estimate); // x_hat = x_k^- + K * (z_k - H * x_k^-)
    kf_error = (1 - kf_gain) * predict_error;     // P = (1 - K) * P_k^-

    ROS_INFO_THROTTLE(1.0, "G = %.4f; Es = %.4f; Er = %.4f", kf_gain, kf_estimate, kf_error);

    // 更新滤波后的全局变量
    depth_filtered = kf_estimate;

    // 发布目标点坐标
    geometry_msgs::PointStamped target_msg;
    target_msg.header.stamp = ros::Time::now();
    target_msg.header.frame_id = "base_link";
    target_msg.point.x = depth_filtered;
    target_msg.point.y = 0;
    target_msg.point.z = 0;
    target_pub.publish(target_msg);
}

void depthCallback(const sensor_msgs::ImageConstPtr& depth_msg){
    // 转换为 OpenCV 格式
    cv_bridge::CvImagePtr cv_depth_ptr;
    cv_depth_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
    cv::Mat depth_image = cv_depth_ptr->image;
    cv::resize(depth_image, depth_image, cv::Size(IMG_WIDTH, IMG_HEIGHT));

    double minVal;
    double maxVal;
    cv::Point minLoc, maxLoc;

    // 使用掩码排除掉无效点
    cv::Mat col = depth_image.col(LINE_X);
    cv::Mat col_mask = (col != 0);

    // 用 minMaxLoc 找出最小值的位置
    cv::minMaxLoc(col, &minVal, &maxVal, &minLoc, &maxLoc, col_mask);
    depth = minVal;

    if (depth < 1000 || depth > 3500) return;

    depth_history.push_back(depth);
    if (depth_history.size() > DEPTH_BUFFER_SIZE) {
        depth_history.erase(depth_history.begin());
    }

    int sum = 0;
    double normal_sum = 0;
    for (int i = 0; i < depth_history.size(); i++) {
        sum += depth_history.at(i);
    }
    double mean = sum / DEPTH_BUFFER_SIZE;
    for (int i = 0; i < depth_history.size(); i++) {
        normal_sum += (depth_history.at(i) - mean) * (depth_history.at(i) - mean);
    }
    // depth = sum / DEPTH_BUFFER_SIZE;
    normal_sum /= DEPTH_BUFFER_SIZE;
    ROS_INFO_THROTTLE(1.0, "normal = %.4f", normal_sum);
    // ROS_INFO("%d", depth);

    kalman_update(depth);
    // ROS_INFO("Minimum depth at x=%d is %.2f mm at y=%d", LINE_X, minVal, minLoc.y);
}

// 图传部分
cv::Mat frame;
std::mutex frame_mutex;

cv::Scalar RED(0, 0, 255), GREEN(0, 255, 0), BLUE(255, 0, 0);

void imageCallback(const sensor_msgs::ImageConstPtr& rgb_msg){
    cv_bridge::CvImagePtr cv_rgb_ptr;
    cv_rgb_ptr = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat rgb_image = cv_rgb_ptr->image;
    // cv::resize(rgb_image, rgb_image, cv::Size(IMG_WIDTH, IMG_HEIGHT));
    
    // int x = rgb_image.cols / 2 + 120;
    std::string depth_info = std::to_string(depth_filtered) + " / " + std::to_string(depth);
    cv::putText(rgb_image, depth_info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2);
    cv::line(rgb_image, cv::Point(LINE_X, 0), cv::Point(LINE_X, rgb_image.rows), depth < 3500? GREEN: RED, 2);
    
    std::lock_guard<std::mutex> lock(frame_mutex);
    frame = rgb_image.clone();
}

// 将 OpenCV 图像编码为 JPEG 并返回
std::vector<uchar> encode_frame() {
    std::lock_guard<std::mutex> lock(frame_mutex);
    std::vector<uchar> buffer;
    if (!frame.empty()) {
        cv::imencode(".jpg", frame, buffer);
    }
    return buffer;
}

// HTTP 回调函数
int request_handler(void *cls, struct MHD_Connection *connection,
                    const char *url, const char *method, const char *version,
                    const char *upload_data, size_t *upload_data_size, void **ptr) {
    if (std::string(method) != "GET") {
        return MHD_NO;
    }

    // 编码图像
    std::vector<uchar> jpeg_data = encode_frame();
    if (jpeg_data.empty()) {
        return MHD_NO;
    }

    // 创建 HTTP 响应
    struct MHD_Response *response = MHD_create_response_from_buffer(
        jpeg_data.size(), jpeg_data.data(), MHD_RESPMEM_MUST_COPY);
    MHD_add_response_header(response, "Content-Type", "image/jpeg");

    int ret = MHD_queue_response(connection, MHD_HTTP_OK, response);
    MHD_destroy_response(response);
    return ret;
}

// 启动 HTTP 服务器
void start_http_server() {
    struct MHD_Daemon *daemon = MHD_start_daemon(
        MHD_USE_SELECT_INTERNALLY, PORT, NULL, NULL, &request_handler, NULL,
        MHD_OPTION_END);
    if (daemon == NULL) {
        std::cerr << "无法启动 HTTP 服务器" << std::endl;
        return;
    }
    std::cout << "访问 http://ip:2777:"<< std::endl;

    // 保持服务器运行
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    MHD_stop_daemon(daemon);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "depth_localization");
    ros::NodeHandle nh;

    std::thread server_thread(start_http_server);
    
    ros::Subscriber image_sub = nh.subscribe("/camera/color/image_raw", 1, imageCallback);
    ros::Subscriber depth_sub = nh.subscribe("/camera/depth/image_raw", 1, depthCallback);
    target_pub = nh.advertise<geometry_msgs::PointStamped>("/target", 1);
    
    ROS_INFO("Depth localization initialized successfully");
    ros::spin();
    
    return 0;
}
