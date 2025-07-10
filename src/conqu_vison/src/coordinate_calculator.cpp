#include "conqu_vison/coordinate_calculator.h"  

CoordinateCalculator::CoordinateCalculator() : 
    private_nh_("~"), 
    first_measurement_(true),
    smoothing_factor_(0.7),
    tf_listener_(tf_buffer_)  // 初始化TF监听器
{
    // 加载参数
    loadParameters();
    
    // 订阅篮框位姿
    hoop_pose_sub_ = nh_.subscribe("/hoop_pose", 1, 
                                  &CoordinateCalculator::hoopPoseCallback, this);
    
    // 发布计算的距离信息
    distance_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("/hoop_distance", 1);
    distance_vector_pub_ = nh_.advertise<geometry_msgs::Vector3Stamped>("/hoop_distance_vector", 1);
    adjusted_pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/hoop_adjusted_pose", 1);
    distance_marker_pub_ = nh_.advertise<visualization_msgs::Marker>("/distance_marker", 1);
    transformed_pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/hoop_pose_transformed", 1);
    
    ROS_INFO("Coordinate calculator initialized with TF support");
}

void CoordinateCalculator::loadParameters()
{
    // 加载篮框几何参数
    nh_.param<double>("/hoop/rim_height", hoop_rim_height_, 2.43);
    
    // 加载相机参数
    nh_.param<double>("/camera/height", camera_height_, 1.02);
    
    // 计算篮筐在世界坐标系中的预期高度
    expected_rim_world_z_ = hoop_rim_height_;
    
    // 是否使用篮筐高度校正
    private_nh_.param<bool>("use_rim_height_correction", use_rim_height_correction_, true);
    private_nh_.param<double>("rim_z_offset", rim_z_offset_, 0.10);
    
    // 获取目标坐标系参数，默认为"base_link"
    private_nh_.param<std::string>("target_frame", target_frame_, "base_link");
    
    ROS_INFO("Parameters loaded:");
    ROS_INFO("Hoop rim height: %.2f m", hoop_rim_height_);
    ROS_INFO("Camera height: %.2f m", camera_height_);
    ROS_INFO("Use rim height correction: %s", use_rim_height_correction_ ? "true" : "false");
    ROS_INFO("Rim Z offset from backboard center: %.2f m", rim_z_offset_);
    ROS_INFO("Target frame for transformation: %s", target_frame_.c_str());
}

void CoordinateCalculator::hoopPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& pose_msg)
{
    // 将位姿从相机坐标系转换到目标坐标系
    geometry_msgs::PoseStamped transformed_pose;
    if (transformPose(*pose_msg, transformed_pose, target_frame_)) {
        // 发布转换后的位姿
        transformed_pose_pub_.publish(transformed_pose);
        
        ROS_INFO_THROTTLE(1.0, "Transformed hoop position: [%.2f, %.2f, %.2f] in %s frame", 
                transformed_pose.pose.position.x,
                transformed_pose.pose.position.y,
                transformed_pose.pose.position.z,
                target_frame_.c_str());
    } else {
        // 如果转换失败，可以选择使用上一次有效的转换结果或者跳过本次处理
        return;
    }
    

    // 获取篮框在相机坐标系中的位置 (使用原始位姿进行距离计算)
    double hoop_x = pose_msg->pose.position.x;
    double hoop_y = pose_msg->pose.position.y;
    double hoop_z = pose_msg->pose.position.z;
    
    // 调整z坐标以获取篮筐中心的位置（如果启用校正）
    double rim_z = hoop_z;
    if (use_rim_height_correction_) {
        rim_z = hoop_z - rim_z_offset_;  // 假设篮筐在篮板中心下方
    }
    
    // 计算水平距离（相机到篮框的直线距离）
    double horizontal_distance = std::sqrt(hoop_x * hoop_x + hoop_y * hoop_y);
    
    // 计算实际的三维距离
    double distance_3d = std::sqrt(hoop_x * hoop_x + 
                                  hoop_y * hoop_y + 
                                  hoop_z * hoop_z);
    
    // 计算视角（仰角）
    double elevation_angle = std::atan2(hoop_z, horizontal_distance) * 180.0 / M_PI;
    
    // 平滑处理
    if (first_measurement_) {
        last_distance_x_ = hoop_x;
        last_distance_y_ = hoop_y;
        last_distance_z_ = hoop_z;
        first_measurement_ = false;
    } else {
        // 指数移动平均滤波
        hoop_x = smoothing_factor_ * hoop_x + (1 - smoothing_factor_) * last_distance_x_;
        hoop_y = smoothing_factor_ * hoop_y + (1 - smoothing_factor_) * last_distance_y_;
        hoop_z = smoothing_factor_ * hoop_z + (1 - smoothing_factor_) * last_distance_z_;
        
        // 更新上一次的值
        last_distance_x_ = hoop_x;
        last_distance_y_ = hoop_y;
        last_distance_z_ = hoop_z;
        
        // 重新计算距离
        horizontal_distance = std::sqrt(hoop_x * hoop_x + hoop_y * hoop_y);
        distance_3d = std::sqrt(hoop_x * hoop_x + 
                               hoop_y * hoop_y + 
                               hoop_z * hoop_z);
        elevation_angle = std::atan2(hoop_z, horizontal_distance) * 180.0 / M_PI;
    }
    
    // 发布距离数据
    std_msgs::Float64MultiArray distance_msg;
    distance_msg.data.push_back(hoop_x);       // 前进/后退方向距离 (x轴)
    distance_msg.data.push_back(hoop_y);       // 左/右方向距离 (y轴)
    distance_msg.data.push_back(hoop_z);       // 高度方向距离 (z轴)
    distance_msg.data.push_back(horizontal_distance); // 水平距离
    distance_msg.data.push_back(distance_3d);       // 3D距离
    distance_msg.data.push_back(elevation_angle);   // 仰角 (度)
    distance_pub_.publish(distance_msg);
    
    // 发布距离向量
    geometry_msgs::Vector3Stamped distance_vector;
    distance_vector.header = pose_msg->header;
    distance_vector.vector.x = hoop_x;
    distance_vector.vector.y = hoop_y;
    distance_vector.vector.z = hoop_z;
    distance_vector_pub_.publish(distance_vector);
    
    // 计算校正后的篮筐位置
    geometry_msgs::PoseStamped adjusted_pose = *pose_msg;
    if (use_rim_height_correction_) {
        adjusted_pose.pose.position.z = rim_z;
    }
    adjusted_pose_pub_.publish(adjusted_pose);
    
    // 可视化距离
    visualizeDistance(pose_msg->header.stamp, pose_msg->header.frame_id,
                      hoop_x, hoop_y, hoop_z);
    
    // 打印距离信息
    ROS_INFO("Distance to hoop: [%.2f, %.2f, %.2f] meters", 
            hoop_x, hoop_y, hoop_z);
    ROS_INFO("Horizontal distance: %.2f m, 3D distance: %.2f m, Elevation angle: %.2f degrees", 
            horizontal_distance, distance_3d, elevation_angle);
}

void CoordinateCalculator::visualizeDistance(const ros::Time& stamp, const std::string& frame_id,
                      double x, double y, double z)
{
    visualization_msgs::Marker marker;
    marker.header.stamp = stamp;
    marker.header.frame_id = frame_id;
    marker.ns = "distance";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::ARROW;
    marker.action = visualization_msgs::Marker::ADD;
    
    // 箭头起点在原点（相机位置）
    marker.points.resize(2);
    marker.points[0].x = 0;
    marker.points[0].y = 0;
    marker.points[0].z = 0;
    
    // 箭头终点指向篮框
    marker.points[1].x = x;
    marker.points[1].y = y;
    marker.points[1].z = z;
    
    // 设置箭头属性
    marker.scale.x = 0.02;  // 箭杆直径
    marker.scale.y = 0.04;  // 箭头直径
    marker.scale.z = 0.1;   // 箭头长度
    
    // 蓝色
    marker.color.r = 0.0;
    marker.color.g = 0.5;
    marker.color.b = 1.0;
    marker.color.a = 1.0;
    
    marker.lifetime = ros::Duration(0.1);  // 100ms
    
    distance_marker_pub_.publish(marker);
}

bool CoordinateCalculator::transformPose(const geometry_msgs::PoseStamped& input_pose, 
                                         geometry_msgs::PoseStamped& output_pose,
                                         const std::string& target_frame)
{
    try {
        // 设置超时时间，这里使用ROS参数服务器中的值或默认值
        double timeout;
        private_nh_.param<double>("tf_timeout", timeout, 0.5);
        
        // 检查是否可以进行转换
        if(!tf_buffer_.canTransform(target_frame, input_pose.header.frame_id, 
                                   input_pose.header.stamp, ros::Duration(timeout))) {
            ROS_WARN_THROTTLE(1.0, "Cannot transform from %s to %s, waiting for transform...", 
                    input_pose.header.frame_id.c_str(), target_frame.c_str());
            return false;
        }
        
        // 执行转换
        tf_buffer_.transform(input_pose, output_pose, target_frame);
        
        ROS_DEBUG("Successfully transformed pose from %s to %s", 
                input_pose.header.frame_id.c_str(), target_frame.c_str());
        return true;
    }
    catch (tf2::TransformException &ex) {
        ROS_WARN_THROTTLE(1.0, "Transform failed: %s", ex.what());
        return false;
    }
}