#include <opencv2/opencv.hpp>
#include <microhttpd.h>
#include <iostream>
#include <vector>
#include <mutex>
#include <thread>
#include <ctime>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <chrono> // 添加chrono头文件用于毫秒级时间戳

#define PORT 8080

cv::Mat frame;
std::mutex frame_mutex;

// 图像处理函数：裁剪、添加文本等
cv::Mat process_image(
    const cv::Mat& input_image
    // bool crop = false, 
    // const cv::Rect& crop_region = cv::Rect(0, 0, 0, 0)
){
    
    cv::Mat processed = input_image.clone();
    
    // 裁剪图像
    // if (crop && crop_region.width > 0 && crop_region.height > 0 &&
    //     crop_region.x >= 0 && crop_region.y >= 0 &&
    //     crop_region.x + crop_region.width <= input_image.cols &&
    //     crop_region.y + crop_region.height <= input_image.rows) {
    //     processed = processed(crop_region);
    // }
    
    // 添加文本
    // if (add_text && !text.empty()) {
    //     cv::putText(processed, text, text_position, 
    //                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    // }
    
    // 在中央从上到下画一条红线
    int center_x = processed.cols / 2 + 120;
    cv::line(processed, cv::Point(center_x, 0), cv::Point(center_x, processed.rows), cv::Scalar(0, 0, 255), 2);
    // printf("%d", processed.cols);

    return processed;
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

int main() {
    // 打开摄像头
    cv::VideoCapture cap(6);
    if (!cap.isOpened()) {
        std::cerr << "无法打开摄像头" << std::endl;
        return -1;
    }

    // 启动 HTTP 服务器线程
    std::thread server_thread(start_http_server);
    

    // 捕获摄像头画面
    while (true) {
        cv::Mat temp_frame;
        cap >> temp_frame;
        if (temp_frame.empty()) {
            std::cerr << "无法读取摄像头帧" << std::endl;
            break;
        }

        // 处理图像 - 使用毫秒级时间戳
        auto now = std::chrono::system_clock::now();
        auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        std::string timestamp = "time: " + std::to_string(millis);
        // temp_frame = process_image(temp_frame, true, cv::Rect((1920-512)/2, 712, 712, 712), true, timestamp);
        // 添加中央红线
        temp_frame = process_image(temp_frame);
        // temp_frame = process_image(temp_frame,false , cv::Rect(), true, timestamp);
        // 更新全局帧
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            frame = temp_frame.clone();
        }
        

        // 显示画面（可选）
        // cv::imshow("摄像头画面", temp_frame);
        // if (cv::waitKey(1) == 27) { // 按下 ESC 键退出
        //     break;
        // }
    }
    cap.release();
    server_thread.join();
    return 0;
}