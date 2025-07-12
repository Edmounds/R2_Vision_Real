#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <visualization_msgs/Marker.h> 
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <cmath>
// 添加必要的头文件
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>

class BasketDetector
{
public:
  BasketDetector() : tf_buffer_(ros::Duration(10)), tf_listener_(tf_buffer_)
  {
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");
    
    // 从参数服务器读取篮板参数
    nh_private.param<float>("/backboard_detection/basket_height", basket_height_, 2.285);
    nh_private.param<float>("/backboard_detection/basket_length", basket_length_, 1.8);
    nh_private.param<float>("/backboard_detection/basket_width", basket_width_, 0.05);
    nh_private.param<float>("/backboard_detection/height_tolerance", height_tolerance_, 0.1);
    nh_private.param<float>("/backboard_detection/size_tolerance", size_tolerance_, 0.2);
    
    // 相机内参参数
    nh_private.param<double>("/camera/fx", fx_, 460.587);
    nh_private.param<double>("/camera/fy", fy_, 460.748);
    nh_private.param<double>("/camera/cx", cx_, 327.796);
    nh_private.param<double>("/camera/cy", cy_, 243.640);
    
    // 添加深度图像处理相关参数
    nh_private.param<int>("/depth_processing/median_blur_ksize", median_blur_ksize_, 5);
    nh_private.param<double>("/depth_processing/depth_threshold", depth_threshold_, 150.0);
    nh_private.param<double>("/depth_processing/canny_threshold1", canny_threshold1_, 50.0);
    nh_private.param<double>("/depth_processing/canny_threshold2", canny_threshold2_, 150.0);
    nh_private.param<double>("/depth_processing/min_aspect_ratio", min_aspect_ratio_, 1.5);
    nh_private.param<double>("/depth_processing/max_aspect_ratio", max_aspect_ratio_, 3.0);
    nh_private.param<double>("/depth_processing/min_area", min_area_, 1000.0);
    
    // 打印读取的参数值，用于调试
    ROS_INFO("Loaded parameters:");
    ROS_INFO("  basket_height: %.3f", basket_height_);
    ROS_INFO("  basket_length: %.3f", basket_length_);
    ROS_INFO("  basket_width: %.3f", basket_width_);
    ROS_INFO("  height_tolerance: %.3f", height_tolerance_);
    ROS_INFO("  size_tolerance: %.3f", size_tolerance_);
 
    // 正确订阅深度图像
    depth_sub_ = nh.subscribe("/camera/depth/image_raw", 1, &BasketDetector::depthImageCallback, this);
    
    // 发布各处理阶段的图像用于调试
    raw_depth_pub_ = nh.advertise<sensor_msgs::Image>("/basket_detection/raw_depth", 1);
    filtered_depth_pub_ = nh.advertise<sensor_msgs::Image>("/basket_detection/filtered_depth", 1);
    thresholded_pub_ = nh.advertise<sensor_msgs::Image>("/basket_detection/thresholded", 1);
    edges_pub_ = nh.advertise<sensor_msgs::Image>("/basket_detection/edges", 1);
    contour_pub_ = nh.advertise<sensor_msgs::Image>("/basket_detection/contour", 1);
    
    // 发布处理后的图像和检测结果
    processed_depth_pub_ = nh.advertise<sensor_msgs::Image>("/basket_detection/processed_depth", 1);
    backboard_pose_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/basket_detection/backboard_pose", 1);
    marker_pub_ = nh.advertise<visualization_msgs::Marker>("/basket_detection/backboard_marker", 1);
  }

  // 深度图像处理回调函数
  void depthImageCallback(const sensor_msgs::ImageConstPtr& depth_msg)
  {
    try
    {
      // 将ROS图像消息转换为OpenCV格式
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
      cv::Mat depth_image = cv_ptr->image;
      
      // 发布原始深度图像用于调试
      cv_bridge::CvImage raw_depth_msg;
      raw_depth_msg.header = depth_msg->header;
      raw_depth_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
      raw_depth_msg.image = depth_image;
      raw_depth_pub_.publish(raw_depth_msg.toImageMsg());
      
      // 1. 深度图去噪处理
      cv::Mat filtered_depth;
      
      // // 使用中值滤波去除噪声
      // cv::medianBlur(depth_image, filtered_depth, median_blur_ksize_);
      // ROS_DEBUG("Applied median blur with kernel size %d", median_blur_ksize_);
      
      // 另一种选择: 双边滤波（保留边缘的同时平滑区域）
      cv::Mat bilateral_filtered;
      cv::bilateralFilter(depth_image, bilateral_filtered, 9, 75, 75);
      filtered_depth = bilateral_filtered;
      
      // 发布滤波后的深度图像
      cv_bridge::CvImage filtered_msg;
      filtered_msg.header = depth_msg->header;
      filtered_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
      filtered_msg.image = filtered_depth;
      filtered_depth_pub_.publish(filtered_msg.toImageMsg());
      
      // 2. 篮板区域分割
      cv::Mat thresholded;
      cv::threshold(filtered_depth, thresholded, depth_threshold_, 255, cv::THRESH_BINARY);
      ROS_DEBUG("Applied threshold with value %.1f", depth_threshold_);
      
      // 发布二值化图像
      cv::Mat thresh_vis;
      thresholded.convertTo(thresh_vis, CV_8UC1);
      cv_bridge::CvImage thresh_msg;
      thresh_msg.header = depth_msg->header;
      thresh_msg.encoding = sensor_msgs::image_encodings::MONO8;
      thresh_msg.image = thresh_vis;
      thresholded_pub_.publish(thresh_msg.toImageMsg());
      
      // 转换为8位图像用于边缘检测
      cv::Mat depth_8bit;
      thresholded.convertTo(depth_8bit, CV_8U);
      
      // 应用Canny边缘检测
      cv::Mat edges;
      cv::Canny(depth_8bit, edges, canny_threshold1_, canny_threshold2_);
      ROS_DEBUG("Applied Canny edge detection with thresholds %.1f, %.1f", 
                canny_threshold1_, canny_threshold2_);
                
      // 发布边缘检测图像
      cv_bridge::CvImage edges_msg;
      edges_msg.header = depth_msg->header;
      edges_msg.encoding = sensor_msgs::image_encodings::MONO8;
      edges_msg.image = edges;
      edges_pub_.publish(edges_msg.toImageMsg());
      
      // 3. 轮廓提取与筛选
      std::vector<std::vector<cv::Point>> contours;
      std::vector<cv::Vec4i> hierarchy;
      cv::findContours(edges, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
      
      // 筛选出符合篮板特征的轮廓
      std::vector<cv::Point> selected_contour;
      cv::Rect selected_rect;
      bool found_backboard = false;
      
      // 创建轮廓可视化图像
      cv::Mat contour_img = cv::Mat::zeros(edges.size(), CV_8UC3);
      // 绘制所有轮廓
      for (size_t i = 0; i < contours.size(); i++) {
        cv::Scalar color = cv::Scalar(0, 0, 255); // 红色：未选择的轮廓
        cv::drawContours(contour_img, contours, i, color, 2);
      }
      
      for (const auto& contour : contours)
      {
        cv::Rect rect = cv::boundingRect(contour);
        double aspect_ratio = rect.width / static_cast<double>(rect.height);
        double area = rect.width * rect.height;
        
        if (min_aspect_ratio_ < aspect_ratio && aspect_ratio < max_aspect_ratio_ && area > min_area_)
        {
          selected_contour = contour;
          selected_rect = rect;
          found_backboard = true;
          
          // 在轮廓图像上绘制选择的轮廓（用绿色标记）
          cv::drawContours(contour_img, std::vector<std::vector<cv::Point>>{contour}, 0, cv::Scalar(0, 255, 0), 3);
          cv::rectangle(contour_img, selected_rect, cv::Scalar(0, 255, 0), 2);
          
          // 如果需要进一步筛选，可以添加更多条件
          break;  // 选取第一个符合条件的轮廓
        }
      }
      
      // 发布轮廓图像
      cv_bridge::CvImage contour_msg;
      contour_msg.header = depth_msg->header;
      contour_msg.encoding = sensor_msgs::image_encodings::BGR8;
      contour_msg.image = contour_img;
      contour_pub_.publish(contour_msg.toImageMsg());
      
      // 4. 坐标转换 - 如果找到了篮板
      std::vector<cv::Point> corner_points; // 将变量定义移到这里，使其对整个函数可见
      
      if (found_backboard)
      {
        ROS_INFO("Found potential backboard: width=%d, height=%d, aspect_ratio=%.2f", 
                 selected_rect.width, selected_rect.height, 
                 selected_rect.width / static_cast<double>(selected_rect.height));
        
        // 提取篮板四个角点
        std::vector<cv::Point3d> backboard_corners_3d;
        
     
        corner_points = { // 修改为赋值而非声明
          cv::Point(selected_rect.x, selected_rect.y),                           // 左上
          cv::Point(selected_rect.x + selected_rect.width, selected_rect.y),     // 右上
          cv::Point(selected_rect.x, selected_rect.y + selected_rect.height),    // 左下
          cv::Point(selected_rect.x + selected_rect.width, selected_rect.y + selected_rect.height) // 右下
        };
        
        for (const auto& point : corner_points)
        {
          int u = point.x;
          int v = point.y;
          
          // 确保坐标在图像范围内
          if (u >= 0 && u < filtered_depth.cols && v >= 0 && v < filtered_depth.rows)
          {
            try {
              // 获取该点的深度值
              float z = filtered_depth.at<float>(v, u);
              
              // 深度值有效时进行转换
              if (std::isfinite(z) && z > 0)
              {
                // 像素坐标转换为3D世界坐标
                double x = (u - cx_) * z / fx_;
                double y = (v - cy_) * z / fy_;
                
                backboard_corners_3d.emplace_back(x, y, z);
                ROS_INFO("Corner 3D coordinate: (%.2f, %.2f, %.2f) m", x, y, z);
              }
              else {
                ROS_DEBUG("Invalid depth value at point (%d, %d): %.2f", u, v, z);
              }
            } catch (const std::exception& e) {
              ROS_ERROR("Exception while processing depth at (%d, %d): %s", u, v, e.what());
            }
          }
          else {
            ROS_DEBUG("Point (%d, %d) out of bounds (image: %dx%d)", 
                      u, v, filtered_depth.cols, filtered_depth.rows);
          }
        }
        
        // 如果成功提取到足够的角点，计算篮板中心位置
        if (backboard_corners_3d.size() >= 3)
        {
          // 计算篮板中心点
          cv::Point3d center(0, 0, 0);
          for (const auto& point : backboard_corners_3d)
          {
            center.x += point.x;
            center.y += point.y;
            center.z += point.z;
          }
          center.x /= backboard_corners_3d.size();
          center.y /= backboard_corners_3d.size();
          center.z /= backboard_corners_3d.size();
          
          // 发布篮板位置
          publishBackboardPose(center, depth_msg->header.frame_id);
          
          // 发布可视化标记
          publishBackboardMarker(center, backboard_corners_3d, depth_msg->header.frame_id);
        }
      }
      else
      {
        ROS_INFO_THROTTLE(1.0, "No suitable backboard contour found");
      }
      
      // 发布最终处理后的图像（用于调试）
      // 创建一个彩色图像展示最终结果
      cv::Mat final_result;
      if (found_backboard) {
        cv::cvtColor(edges, final_result, cv::COLOR_GRAY2BGR);
        // 在最终结果上绘制检测到的篮板位置
        cv::rectangle(final_result, selected_rect, cv::Scalar(0, 255, 0), 2);
        
        // 绘制四个角点
        for (const auto& point : corner_points) {
          cv::circle(final_result, point, 5, cv::Scalar(0, 0, 255), -1);
        }
      } else {
        cv::cvtColor(edges, final_result, cv::COLOR_GRAY2BGR);
      }
      
      cv_bridge::CvImage processed_msg;
      processed_msg.header = depth_msg->header;
      processed_msg.encoding = sensor_msgs::image_encodings::BGR8;
      processed_msg.image = final_result;
      processed_depth_pub_.publish(processed_msg.toImageMsg());
      
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
    }
  }

private:
  // 发布篮板位置
  void publishBackboardPose(const cv::Point3d& center, const std::string& frame_id)
  {
    geometry_msgs::PoseStamped pose;
    pose.header.stamp = ros::Time::now();
    pose.header.frame_id = frame_id;
    
    pose.pose.position.x = center.x;
    pose.pose.position.y = center.y;
    pose.pose.position.z = center.z;
    
    // 默认方向（假设篮板面向相机）
    pose.pose.orientation.w = 1.0;
    
    backboard_pose_pub_.publish(pose);
  }
  
  // 发布可视化标记
  void publishBackboardMarker(const cv::Point3d& center, 
                             const std::vector<cv::Point3d>& corners,
                             const std::string& frame_id)
  {
    visualization_msgs::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = ros::Time::now();
    marker.ns = "backboard";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::CUBE;
    marker.action = visualization_msgs::Marker::ADD;
    
    // 设置位置
    marker.pose.position.x = center.x;
    marker.pose.position.y = center.y;
    marker.pose.position.z = center.z;
    
    // 默认方向
    marker.pose.orientation.w = 1.0;
    
    // 估计尺寸
    if (corners.size() >= 4)
    {
      // 简单估计宽度和高度
      double width = cv::norm(corners[0] - corners[1]);
      double height = cv::norm(corners[0] - corners[2]);
      
      marker.scale.x = 0.05;  // 厚度
      marker.scale.y = width;
      marker.scale.z = height;
    }
    else
    {
      // 默认尺寸
      marker.scale.x = basket_width_;
      marker.scale.y = basket_length_;
      marker.scale.z = 1.0;  // 默认高度
    }
    
    // 设置颜色
    marker.color.r = 1.0;
    marker.color.g = 1.0;
    marker.color.b = 1.0;
    marker.color.a = 0.8;
    
    marker.lifetime = ros::Duration(0.5);  // 0.5秒
    
    marker_pub_.publish(marker);
  }

  // ROS相关
  ros::NodeHandle nh_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  ros::Subscriber depth_sub_;
  
  // 用于调试的图像发布器
  ros::Publisher raw_depth_pub_;
  ros::Publisher filtered_depth_pub_;
  ros::Publisher thresholded_pub_;
  ros::Publisher edges_pub_;
  ros::Publisher contour_pub_;
  ros::Publisher processed_depth_pub_;
  
  // 篮板位姿和标记发布器
  ros::Publisher backboard_pose_pub_;
  ros::Publisher marker_pub_;
  
  // 篮板参数
  float basket_height_;
  float basket_length_;
  float basket_width_;
  float height_tolerance_;
  float size_tolerance_;
  
  // 相机内参
  double fx_, fy_, cx_, cy_;
  
  // 图像处理参数
  int median_blur_ksize_;
  double depth_threshold_;
  double canny_threshold1_;
  double canny_threshold2_;
  double min_aspect_ratio_;
  double max_aspect_ratio_;
  double min_area_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "backboard_detector");
  BasketDetector detector;
  ros::spin();
  return 0;
}

