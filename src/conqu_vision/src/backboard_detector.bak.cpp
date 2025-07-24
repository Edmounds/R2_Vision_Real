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



class BackboardDetector
{
private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    
    // 消息过滤器
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub_;
    message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
    typedef message_filters::Synchronizer<SyncPolicy> Synchronizer;
    boost::shared_ptr<Synchronizer> sync_;
    
    image_transport::Publisher image_pub_;
    image_transport::Publisher cropped_depth_pub_;  // 裁剪后的深度图发布器
    
    // 深度图信息
    int depth_width_ = 0;
    int depth_height_ = 0;
    // OpenVINO相关
    ov::Core core_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;
    
    // 模型参数
    std::string model_path_;
    int input_width_;
    int input_height_;
    float confidence_threshold_;
    std::vector<std::string> class_names_;
    
    // 添加图像处理参数
    int canny_threshold1_ = 100;
    int canny_threshold2_ = 200;
    double epsilon_factor_ = 0.02;
    double min_contour_area_ = 100.0;
    
    // 多帧稳定性
    std::vector<cv::Point> last_good_approx_;
    int stable_frames_ = 0;
    ros::Time last_process_time_;
    
    // 霍夫线检测和评分参数
    int hough_threshold_ = 30;         // 霍夫线检测阈值
    double sL_dLU_weight_ = 15.0;      // 左线dLU权重
    double sL_dLD_weight_ = 8.0;       // 左线dLD权重
    double sL_angle_weight_ = 40.0;    // 左线angle权重
    double sR_dRU_weight_ = 15.0;      // 右线dRU权重
    double sR_dRD_weight_ = 8.0;       // 右线dRD权重
    double sR_angle_weight_ = 40.0;    // 右线angle权重
    
    // GUI 窗口名称
    std::string window_name_ = "Backboard Detector Parameters";
    
public:
    BackboardDetector()
        : it_(nh_), last_process_time_(ros::Time::now())
    {
        // 读取参数
        ros::param::param<std::string>("~model_path", model_path_, "/home/rc1/cqc/R2_Real_ws/hoop.onnx");
        ros::param::param<int>("~input_width", input_width_, 640);
        ros::param::param<int>("~input_height", input_height_, 640);
        ros::param::param<float>("~confidence_threshold", confidence_threshold_, 0.7f);
        
        // 初始化图像处理参数
        ros::param::param<int>("~canny_threshold1", canny_threshold1_, 100);
        ros::param::param<int>("~canny_threshold2", canny_threshold2_, 200);
        ros::param::param<double>("~epsilon_factor", epsilon_factor_, 0.02);
        ros::param::param<double>("~min_contour_area", min_contour_area_, 100.0);
        
        // 初始化霍夫线检测和评分参数
        ros::param::param<int>("~hough_threshold", hough_threshold_, 30);
        ros::param::param<double>("~sL_dLU_weight", sL_dLU_weight_, 15.0);
        ros::param::param<double>("~sL_dLD_weight", sL_dLD_weight_, 8.0);
        ros::param::param<double>("~sL_angle_weight", sL_angle_weight_, 40.0);
        ros::param::param<double>("~sR_dRU_weight", sR_dRU_weight_, 15.0);
        ros::param::param<double>("~sR_dRD_weight", sR_dRD_weight_, 8.0);
        ros::param::param<double>("~sR_angle_weight", sR_angle_weight_, 40.0);
        
        // 初始化类别名称
        class_names_ = {"basketball_hoop"};
        
        // 设置订阅
        rgb_sub_.subscribe(nh_, "/camera/color/image_raw", 1);
        depth_sub_.subscribe(nh_, "/camera/depth/image_raw", 1);
        
        // 使用近似时间同步策略
        sync_.reset(new Synchronizer(SyncPolicy(10), rgb_sub_, depth_sub_));
        sync_->registerCallback(boost::bind(&BackboardDetector::imageCbWithDepth, this, _1, _2));
        
        // 发布检测结果话题
        image_pub_ = it_.advertise("/detection/result", 1);
        // 发布裁剪后的深度图话题
        cropped_depth_pub_ = it_.advertise("/detection/cropped_depth", 1);
        
        // 初始化OpenVINO
        initOpenVINO();
        
        // 创建GUI控制窗口和滑动条
        createGUI();
        
        ROS_INFO("Backboard detector initialized successfully");
    }

private:
    void initOpenVINO()
    {
        try {
            ROS_INFO("Loading OpenVINO model: %s", model_path_.c_str());
            
            // 读取模型
            auto model = core_.read_model(model_path_);
            
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
                compiled_model_ = core_.compile_model(model, "GPU", config);
                ROS_INFO("Using Intel GPU for inference");
            } catch (const std::exception& e) {
                ROS_WARN("Failed to use GPU, falling back to CPU: %s", e.what());
                compiled_model_ = core_.compile_model(model, "CPU");
            }
            
            // 创建推理请求
            infer_request_ = compiled_model_.create_infer_request();
            
        } catch (const std::exception& e) {
            ROS_ERROR("Failed to initialize OpenVINO: %s", e.what());
            ros::shutdown();
        }
    }
    
    // 创建GUI和滑动条
    void createGUI()
    {
        // 创建窗口
        cv::namedWindow(window_name_, cv::WINDOW_AUTOSIZE);
        
        // 创建一个初始的空白图像以确保窗口显示
        cv::Mat gui_image = cv::Mat(300, 500, CV_8UC3, cv::Scalar(240, 240, 240));
        cv::putText(gui_image, "Basketball Hoop Detector Parameters", cv::Point(10, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
        cv::putText(gui_image, "Adjust parameters using sliders", cv::Point(10, 60), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
        
        // 显示图像以确保窗口创建
        cv::imshow(window_name_, gui_image);
        cv::waitKey(1);  // 处理窗口事件
        
        // 创建霍夫线检测阈值滑动条
        cv::createTrackbar("Hough Threshold", window_name_, &hough_threshold_, 100, onTrackbarStatic, this);
        
        // 创建左线评分权重滑动条
        int sL_dLU_weight_int = static_cast<int>(sL_dLU_weight_ * 10);
        cv::createTrackbar("sL dLU Weight x10", window_name_, &sL_dLU_weight_int, 500, onSL_dLU_TrackbarStatic, this);
        
        int sL_dLD_weight_int = static_cast<int>(sL_dLD_weight_ * 10);
        cv::createTrackbar("sL dLD Weight x10", window_name_, &sL_dLD_weight_int, 300, onSL_dLD_TrackbarStatic, this);
        
        int sL_angle_weight_int = static_cast<int>(sL_angle_weight_ * 10);
        cv::createTrackbar("sL Angle Weight x10", window_name_, &sL_angle_weight_int, 1000, onSL_angle_TrackbarStatic, this);
        
        // 创建右线评分权重滑动条
        int sR_dRU_weight_int = static_cast<int>(sR_dRU_weight_ * 10);
        cv::createTrackbar("sR dRU Weight x10", window_name_, &sR_dRU_weight_int, 500, onSR_dRU_TrackbarStatic, this);
        
        int sR_dRD_weight_int = static_cast<int>(sR_dRD_weight_ * 10);
        cv::createTrackbar("sR dRD Weight x10", window_name_, &sR_dRD_weight_int, 300, onSR_dRD_TrackbarStatic, this);
        
        int sR_angle_weight_int = static_cast<int>(sR_angle_weight_ * 10);
        cv::createTrackbar("sR Angle Weight x10", window_name_, &sR_angle_weight_int, 1000, onSR_angle_TrackbarStatic, this);
        
        // 再次显示窗口以确保滑动条可见
        cv::imshow(window_name_, gui_image);
        cv::waitKey(1);
        
        ROS_INFO("GUI window created with sliders. If not visible, check your display settings.");
    }
    
    // 静态回调函数，将回调转发给对象
    static void onTrackbarStatic(int value, void* userdata) {
        BackboardDetector* detector = static_cast<BackboardDetector*>(userdata);
        detector->onTrackbar(value);
    }
    
    static void onSL_dLU_TrackbarStatic(int value, void* userdata) {
        BackboardDetector* detector = static_cast<BackboardDetector*>(userdata);
        detector->onSL_dLU_Trackbar(value);
    }
    
    static void onSL_dLD_TrackbarStatic(int value, void* userdata) {
        BackboardDetector* detector = static_cast<BackboardDetector*>(userdata);
        detector->onSL_dLD_Trackbar(value);
    }
    
    static void onSL_angle_TrackbarStatic(int value, void* userdata) {
        BackboardDetector* detector = static_cast<BackboardDetector*>(userdata);
        detector->onSL_angle_Trackbar(value);
    }
    
    static void onSR_dRU_TrackbarStatic(int value, void* userdata) {
        BackboardDetector* detector = static_cast<BackboardDetector*>(userdata);
        detector->onSR_dRU_Trackbar(value);
    }
    
    static void onSR_dRD_TrackbarStatic(int value, void* userdata) {
        BackboardDetector* detector = static_cast<BackboardDetector*>(userdata);
        detector->onSR_dRD_Trackbar(value);
    }
    
    static void onSR_angle_TrackbarStatic(int value, void* userdata) {
        BackboardDetector* detector = static_cast<BackboardDetector*>(userdata);
        detector->onSR_angle_Trackbar(value);
    }
    
    // 各滑动条的回调函数
    void onTrackbar(int value) {
        hough_threshold_ = value;
        ROS_INFO("Hough threshold set to: %d", hough_threshold_);
        
        // 更新GUI界面
        cv::setTrackbarPos("Hough Threshold", window_name_, hough_threshold_);
    }
    
    void onSL_dLU_Trackbar(int value) {
        sL_dLU_weight_ = static_cast<double>(value) / 10.0;
        ROS_INFO("sL dLU weight set to: %.1f", sL_dLU_weight_);
        
        // 更新GUI界面
        cv::setTrackbarPos("sL dLU Weight x10", window_name_, value);
    }
    
    void onSL_dLD_Trackbar(int value) {
        sL_dLD_weight_ = static_cast<double>(value) / 10.0;
        ROS_INFO("sL dLD weight set to: %.1f", sL_dLD_weight_);
        
        // 更新GUI界面
        cv::setTrackbarPos("sL dLD Weight x10", window_name_, value);
    }
    
    void onSL_angle_Trackbar(int value) {
        sL_angle_weight_ = static_cast<double>(value) / 10.0;
        ROS_INFO("sL angle weight set to: %.1f", sL_angle_weight_);
        
        // 更新GUI界面
        cv::setTrackbarPos("sL Angle Weight x10", window_name_, value);
    }
    
    void onSR_dRU_Trackbar(int value) {
        sR_dRU_weight_ = static_cast<double>(value) / 10.0;
        ROS_INFO("sR dRU weight set to: %.1f", sR_dRU_weight_);
        
        // 更新GUI界面
        cv::setTrackbarPos("sR dRU Weight x10", window_name_, value);
    }
    
    void onSR_dRD_Trackbar(int value) {
        sR_dRD_weight_ = static_cast<double>(value) / 10.0;
        ROS_INFO("sR dRD weight set to: %.1f", sR_dRD_weight_);
        
        // 更新GUI界面
        cv::setTrackbarPos("sR dRD Weight x10", window_name_, value);
    }
    
    void onSR_angle_Trackbar(int value) {
        sR_angle_weight_ = static_cast<double>(value) / 10.0;
        ROS_INFO("sR angle weight set to: %.1f", sR_angle_weight_);
        
        // 更新GUI界面
        cv::setTrackbarPos("sR Angle Weight x10", window_name_, value);
    }
    
    cv::Mat preprocessImage(const cv::Mat& image)
    {
        cv::Mat rgb_image, resized_image;
        
        // BGR转RGB
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
        
        // 缩放到模型输入尺寸
        cv::resize(rgb_image, resized_image, cv::Size(input_width_, input_height_));
        
        // 转换数据类型并归一化
        cv::Mat float_image;
        resized_image.convertTo(float_image, CV_32F, 1.0/255.0);
        
        return float_image;
    }
    
    struct Detection {
        cv::Rect bbox;
        float confidence;
        int class_id;
    };
    
    std::vector<Detection> postprocessDetections(const ov::Tensor& output_tensor, 
                                               float original_width, float original_height)
    {
        std::vector<Detection> detections;
        
        // 获取输出数据
        const float* data = output_tensor.data<float>();
        
        // YOLOv11输出格式: [batch, num_classes + 4, num_anchors]
        auto shape = output_tensor.get_shape();
        size_t num_classes_plus_4 = shape[1];  // 类别数 + 4个坐标
        size_t num_anchors = shape[2];
        size_t num_classes = num_classes_plus_4 - 4;
        
        // 计算缩放因子
        float scale_x = original_width / input_width_;
        float scale_y = original_height / input_height_;
        
        for (size_t i = 0; i < num_anchors; ++i) {
            // 提取边界框坐标 (中心点格式)
            float x_center = data[i];
            float y_center = data[num_anchors + i];
            float width = data[2 * num_anchors + i];
            float height = data[3 * num_anchors + i];
            
            // 找到最高置信度的类别
            float max_confidence = 0.0f;
            int max_class_id = -1;
            
            for (size_t c = 0; c < num_classes; ++c) {
                float confidence = data[(4 + c) * num_anchors + i];
                if (confidence > max_confidence) {
                    max_confidence = confidence;
                    max_class_id = c;
                }
            }
            
            // 过滤低置信度检测
            if (max_confidence >= confidence_threshold_) {
                // 转换为左上角格式并恢复原始尺寸
                float x1 = (x_center - width / 2) * scale_x;
                float y1 = (y_center - height / 2) * scale_y;
                float x2 = (x_center + width / 2) * scale_x;
                float y2 = (y_center + height / 2) * scale_y;
                
                // 确保坐标在图像范围内
                x1 = std::max(0.0f, std::min(x1, original_width - 1));
                y1 = std::max(0.0f, std::min(y1, original_height - 1));
                x2 = std::max(0.0f, std::min(x2, original_width - 1));
                y2 = std::max(0.0f, std::min(y2, original_height - 1));
                
                Detection det;
                det.bbox = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
                det.confidence = max_confidence;
                det.class_id = max_class_id;
                
                detections.push_back(det);
            }
        }
        
        return detections;
    }
    
    cv::Mat drawDetections(const cv::Mat& image, const std::vector<Detection>& detections)
    {
        cv::Mat result = image.clone();
        
        for (const auto& detection : detections) {
            // 绘制边界框
            cv::rectangle(result, detection.bbox, cv::Scalar(0, 255, 0), 2);
            
            // 准备标签文本
            std::string class_name = detection.class_id < class_names_.size() ? 
                class_names_[detection.class_id] : "Unknown";
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
    
    cv::Rect mapRgbRectToDepth(const cv::Rect& rgb_rect, int rgb_width, int rgb_height)
    {
        if (depth_width_ == 0 || depth_height_ == 0) {
            return rgb_rect;  // 如果还没有深度图信息，直接返回原始框
        }
        
        // 计算比例
        float scale_x = static_cast<float>(depth_width_) / static_cast<float>(rgb_width);
        float scale_y = static_cast<float>(depth_height_) / static_cast<float>(rgb_height);
        
        // 映射矩形
        int width = static_cast<int>(rgb_rect.width * scale_x * 1.05);
        int height = static_cast<int>(rgb_rect.height * scale_y * 1.05);
        int x = static_cast<int>(rgb_rect.x * scale_x - width * 0.025);  // 向左扩展5%
        int y = static_cast<int>(rgb_rect.y * scale_y - height * 0.025);  // 向上扩展5%
        
        // 确保坐标在深度图范围内
        x = std::max(0, std::min(x, depth_width_ - 1));
        y = std::max(0, std::min(y, depth_height_ - 1));
        width = std::min(width, depth_width_ - x);
        height = std::min(height, depth_height_ - y);
        
        return cv::Rect(x, y, width, height);
    }

    void imageCb(const sensor_msgs::ImageConstPtr& msg)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
        
        cv::Mat original_image = cv_ptr->image;
        float original_width = original_image.cols;
        float original_height = original_image.rows;
        
        try {
            // 预处理图像
            cv::Mat preprocessed = preprocessImage(original_image);
            
            // 创建输入张量
            ov::Tensor input_tensor = ov::Tensor(ov::element::f32, {1, 3, static_cast<size_t>(input_height_), static_cast<size_t>(input_width_)});
            float* input_data = input_tensor.data<float>();
            
            // 复制数据到张量 (HWC -> CHW)
            for (int c = 0; c < 3; ++c) {
                for (int h = 0; h < input_height_; ++h) {
                    for (int w = 0; w < input_width_; ++w) {
                        input_data[c * input_height_ * input_width_ + h * input_width_ + w] = 
                            preprocessed.at<cv::Vec3f>(h, w)[c];
                    }
                }
            }
            
            // 设置输入张量
            infer_request_.set_input_tensor(input_tensor);
            
            // 运行推理
            infer_request_.infer();
            
            // 获取输出
            auto output_tensor = infer_request_.get_output_tensor();
            
            // 后处理
            std::vector<Detection> detections = postprocessDetections(output_tensor, 
                original_width, original_height);
            
            
            // 绘制检测结果
            cv::Mat result_image = drawDetections(original_image, detections);
            
            // 发布结果
            cv_bridge::CvImage out_msg;
            out_msg.header = msg->header;
            out_msg.encoding = sensor_msgs::image_encodings::BGR8;
            out_msg.image = result_image;
            
            image_pub_.publish(out_msg.toImageMsg());
            
        } catch (const std::exception& e) {
            ROS_ERROR("Inference failed: %s", e.what());
        }
    }

    void processDetectionsWithDepth(const std::vector<Detection>& detections, 
                              int rgb_width, int rgb_height,
                              const cv::Mat& depth_image,
                              const ros::Time& timestamp)
    {
        // 检查是否有订阅者，如果没有，就不进行处理
        // if (cropped_depth_pub_.getNumSubscribers() == 0 && detections.empty()) {
        //     return;
        // }

        ROS_INFO("Processing detections with depth image");
        
        // 添加帧率控制，最多10Hz处理频率
        if ((ros::Time::now() - last_process_time_).toSec() < 0.1) {
            return;
        }
        last_process_time_ = ros::Time::now();
        
        for (const auto& detection : detections) {
            // 映射检测框到深度图
            cv::Rect depth_rect = mapRgbRectToDepth(detection.bbox, rgb_width, rgb_height);
            
            // 提取该区域的深度图
            cv::Mat roi = depth_image(depth_rect).clone();  // 使用clone()创建一个副本

            // 将16位深度图转换为8位格式，便于进行Canny边缘检测
            cv::Mat roi_8bit;
            // 归一化到0-255范围
            double minVal, maxVal;
            cv::minMaxLoc(roi, &minVal, &maxVal);
            // 避免除以零
            if (maxVal > minVal) {
                roi.convertTo(roi_8bit, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
            } else {
                roi.convertTo(roi_8bit, CV_8U, 0);
            }

            cv::imwrite("/home/rc1/cqc/R2_Real_ws/cropped_depth.png", roi_8bit);

            // 预处理：高斯模糊去噪
            cv::GaussianBlur(roi_8bit, roi_8bit, cv::Size(5, 5), 1.5);
            
            // 预处理：增强对比度
            cv::equalizeHist(roi_8bit, roi_8bit);
            
            // 对8位图像进行边缘检测，参数优化为cv-test.py中的值
            cv::Mat edge;
            cv::Canny(roi_8bit, edge, 10, 50);
            
            // 创建一个用于显示的彩色图像
            cv::Mat display_img;
            cv::cvtColor(edge, display_img, cv::COLOR_GRAY2BGR);
            
            // 使用霍夫变换检测直线，参数优化为cv-test.py中的值
            std::vector<cv::Vec4i> lines;
            cv::HoughLinesP(edge, lines, 2, CV_PI / 90, hough_threshold_, 100, 50);
            
            // 获取图像尺寸
            int height = edge.rows;
            int width = edge.cols;
            
            // 初始化四条边线的端点和评分
            cv::Point L1, L2, R1, R2, U1, U2, D1, D2;
            double sLMax = -1000000, sRMax = -1000000;
            
            // 寻找最优的左右边线
            if (!lines.empty()) {
                for (const auto& line : lines) {
                    int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
                    
                    // 计算线段长度
                    double length = std::sqrt(std::pow(x2 - x1, 2) + std::pow(y2 - y1, 2));
                    
                    if (length < 100 || length > 150) {
                        continue;
                    }
                    
                    // 计算到四个角点的距离
                    double dLU = std::min(
                        std::sqrt(std::pow(x1 - 0, 2) + std::pow(y1 - 0, 2)),
                        std::sqrt(std::pow(x2 - 0, 2) + std::pow(y2 - 0, 2))
                    ); // 左上角
                    
                    double dRU = std::min(
                        std::sqrt(std::pow(x1 - width, 2) + std::pow(y1 - 0, 2)),
                        std::sqrt(std::pow(x2 - width, 2) + std::pow(y2 - 0, 2))
                    ); // 右上角
                    
                    double dLD = std::min(
                        std::sqrt(std::pow(x1 - 0, 2) + std::pow(y1 - height, 2)),
                        std::sqrt(std::pow(x2 - 0, 2) + std::pow(y2 - height, 2))
                    ); // 左下角
                    
                    double dRD = std::min(
                        std::sqrt(std::pow(x1 - width, 2) + std::pow(y1 - height, 2)),
                        std::sqrt(std::pow(x2 - width, 2) + std::pow(y2 - height, 2))
                    ); // 右下角
                    
                    // 计算左边线和右边线的评分，使用滑动条设置的权重
                    double angle = std::abs(std::atan2(abs(y2 - y1), abs(x2 - x1))) * 180.0 / CV_PI;

                    double sL = length - dLU * sL_dLU_weight_ - dLD * sL_dLD_weight_ + angle * sL_angle_weight_;
                    double sR = length - dRU * sR_dRU_weight_ - dRD * sR_dRD_weight_ + angle * sR_angle_weight_;

                    // 更新左边线
                    if (sL > sLMax) {
                        sLMax = sL;
                        L1 = cv::Point(x1, y1);
                        L2 = cv::Point(x2, y2);
                    }
                    
                    // 更新右边线
                    if (sR > sRMax) {
                        sRMax = sR;
                        R1 = cv::Point(x1, y1);
                        R2 = cv::Point(x2, y2);
                    }
                }
                
                // 绘制检测到的左右边线
                if (sLMax > -1000000) {
                    cv::line(display_img, L1, L2, cv::Scalar(255, 0, 0), 2); // 左边线 - 蓝色
                }
                
                if (sRMax > -1000000) {
                    cv::line(display_img, R1, R2, cv::Scalar(0, 255, 0), 2); // 右边线 - 绿色
                }
                
                // 输出调试信息
                ROS_INFO_THROTTLE(1.0, "Left line: (%d,%d)->(%d,%d), Right line: (%d,%d)->(%d,%d)",
                                L1.x, L1.y, L2.x, L2.y, R1.x, R1.y, R2.x, R2.y);
                
                // 如果同时检测到左右边线，可以尝试找出上下交点
                if (sLMax > -1000000 && sRMax > -1000000) {
                    // 这里可以添加计算上下交点的代码，暂时不实现
                }
            }
            
            // 发布处理后的图像
            cv_bridge::CvImage cropped_depth_msg;
            cropped_depth_msg.header.stamp = timestamp;
            cropped_depth_msg.encoding = sensor_msgs::image_encodings::BGR8;
            cropped_depth_msg.image = display_img;
            cropped_depth_pub_.publish(cropped_depth_msg.toImageMsg());
            
            // 处理GUI事件，确保窗口响应
            cv::waitKey(1);
        }
    }
    
    void imageCbWithDepth(const sensor_msgs::ImageConstPtr& rgb_msg,
                     const sensor_msgs::ImageConstPtr& depth_msg)
    {
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
        
        // 更新深度图尺寸
        depth_width_ = depth_image.cols;
        depth_height_ = depth_image.rows;
        
        float rgb_width = rgb_image.cols;
        float rgb_height = rgb_image.rows;
        
        try {
            // 预处理图像
            cv::Mat preprocessed = preprocessImage(rgb_image);
            
            // 创建输入张量
            ov::Tensor input_tensor = ov::Tensor(ov::element::f32, {1, 3, static_cast<size_t>(input_height_), static_cast<size_t>(input_width_)});
            float* input_data = input_tensor.data<float>();
            
            // 复制数据到张量 (HWC -> CHW)
            for (int c = 0; c < 3; ++c) {
                for (int h = 0; h < input_height_; ++h) {
                    for (int w = 0; w < input_width_; ++w) {
                        input_data[c * input_height_ * input_width_ + h * input_width_ + w] = 
                            preprocessed.at<cv::Vec3f>(h, w)[c];
                    }
                }
            }
            
            // 设置输入张量
            infer_request_.set_input_tensor(input_tensor);
            
            // 运行推理
            infer_request_.infer();
            
            // 获取输出
            auto output_tensor = infer_request_.get_output_tensor();
            
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
            
            image_pub_.publish(out_msg.toImageMsg());
            
        } catch (const std::exception& e) {
            ROS_ERROR("Inference failed: %s", e.what());
        }
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "backboard_detector");
    BackboardDetector detector;
    
    // 创建一个定时器用于更新GUI
    ros::NodeHandle nh;
    ros::Timer gui_timer = nh.createTimer(ros::Duration(0.05), // 20Hz
        [](const ros::TimerEvent&) {
            // 使用1ms的等待时间处理GUI事件
            cv::waitKey(1);
        });
    
    ros::spin();
    
    // 销毁所有OpenCV窗口
    cv::destroyAllWindows();
    
    return 0;
}