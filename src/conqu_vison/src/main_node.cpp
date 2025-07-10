
#include <ros/ros.h>
#include <signal.h>
#include <memory>
#include <std_msgs/String.h> 
#include "conqu_vison/hoop_detector.h"  
#include "conqu_vison/coordinate_calculator.h"  
/**
 * @brief 篮框检测和距离计算系统的主控节点
 * 
 * 该类整合了篮框检测器和坐标计算器的功能，
 * 负责初始化系统、加载参数、管理生命周期。
 */
class MainNode
{
private:
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;
    
    // 参数
    std::string camera_topic_;
    double detection_rate_;
    
    // 节点状态
    bool is_running_;
    
    // 子模块
    std::unique_ptr<HoopDetector> hoop_detector_;
    std::unique_ptr<CoordinateCalculator> coordinate_calculator_;
    
    // 定时器
    ros::Timer status_timer_;
    
    // ROS发布器和订阅器
    ros::Publisher system_status_pub_;
    
public:
    MainNode() : private_nh_("~"), is_running_(false)
    {
        ROS_INFO("Initializing Basketball Hoop Detection System...");
        
        // 加载参数
        loadParameters();
        
        // 设置状态发布器
        system_status_pub_ = nh_.advertise<std_msgs::String>("/vision_system/status", 1);
        
        // 创建篮框检测器和坐标计算器
        initializeComponents();
        
        // 设置状态检查定时器
        status_timer_ = nh_.createTimer(ros::Duration(1.0), &MainNode::statusCallback, this);
        
        is_running_ = true;
        publishStatus("System initialized and running");
        
        ROS_INFO("Basketball Hoop Detection System initialized successfully");
    }
    
    ~MainNode()
    {
        shutdown();
    }
    
    void loadParameters()
    {
        // 加载相机话题名称
        private_nh_.param<std::string>("camera_topic", camera_topic_, "/camera/depth/points");
        
        // 加载检测频率
        private_nh_.param<double>("detection_rate", detection_rate_, 30.0);
        
        ROS_INFO("Loaded parameters:");
        ROS_INFO("  Camera topic: %s", camera_topic_.c_str());
        ROS_INFO("  Detection rate: %.1f Hz", detection_rate_);
    }
    
    void initializeComponents()
    {
        ROS_INFO("Creating hoop detector...");
        // 创建篮框检测器
        hoop_detector_ = std::make_unique<HoopDetector>();
        
        ROS_INFO("Creating coordinate calculator...");
        // 创建坐标计算器
        coordinate_calculator_ = std::make_unique<CoordinateCalculator>();
    }
    
    void statusCallback(const ros::TimerEvent& event)
    {
        if (is_running_) {
            // 检查系统状态并发布
            publishStatus("System running normally");
        }
    }
    
    void publishStatus(const std::string& status_msg)
    {
        std_msgs::String msg;
        msg.data = status_msg;
        system_status_pub_.publish(msg);
        ROS_DEBUG("Status: %s", status_msg.c_str());
    }
    
    void shutdown()
    {
        if (is_running_) {
            ROS_INFO("Shutting down Basketball Hoop Detection System...");
            is_running_ = false;
            
            // 停止定时器
            status_timer_.stop();
            
            // 发布关闭状态
            publishStatus("System shutting down");
            
            ROS_INFO("System shutdown complete");
        }
    }
    
    bool isRunning() const
    {
        return is_running_;
    }
};

// 信号处理函数
MainNode* g_main_node = nullptr;

void signalHandler(int sig)
{
    if (g_main_node) {
        g_main_node->shutdown();
    }
    ros::shutdown();
}

int main(int argc, char** argv)
{
    // 初始化ROS节点
    ros::init(argc, argv, "basketball_vision_system", ros::init_options::NoSigintHandler);
    
    // 信号处理
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    // 创建主节点
    MainNode main_node;
    g_main_node = &main_node;
    
    ROS_INFO("Basketball Vision System main node running");
    
    // 检查节点是否正在运行
    while (ros::ok() && main_node.isRunning()) {
        ros::spinOnce();
        // 根据设定的检测率，控制循环频率
        ros::Rate(30).sleep();  // 30Hz
    }
    
    g_main_node = nullptr;
    
    return 0;
}