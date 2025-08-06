#include <ros/ros.h>
#include <conqu_vision/ByteArray.h>
#include <vector>
#include <mutex>
#include <geometry_msgs/PointStamped.h>
#include <std_msgs/UInt8MultiArray.h>
#include <iomanip>
#include <sstream>
#include <algorithm>

// 全局变量保护锁
std::mutex data_mutex;

// 配置参数
const uint8_t FRAME_HEADER = 0xAA;      //帧头
const uint8_t FRAME_FOOTER = 0x55;      //帧尾
const uint8_t PADDING_VALUE = 0x00;     //默认填充
const size_t FIXED_FRAME_SIZE = 10;     // 固定帧长度（可根据需要修改）

// 当前待发送数据
std::vector<uint8_t> x_position_data(2, 0x00); // 存储X坐标数据的 2 字节
std::vector<uint8_t> y_position_data(2, 0x00); // 存储Y坐标数据的 2 字节
std::vector<uint8_t> z_position_data(2, 0x00); // 存储Z坐标数据的 2 字节

// 目标位置数据回调
void targetCallback(const geometry_msgs::PointStamped::ConstPtr& msg) {
    // 处理X坐标
    float x_pos = msg->point.x;
    uint16_t scaled_x = static_cast<uint16_t>(std::abs(x_pos) * 100);  // 取绝对值并放大100倍
    
    // 处理Y坐标
    float y_pos = msg->point.y;
    uint16_t scaled_y = static_cast<uint16_t>(std::abs(y_pos) * 100);
    
    // 处理Z坐标
    float z_pos = msg->point.z;
    uint16_t scaled_z = static_cast<uint16_t>(std::abs(z_pos) * 100);
    
    // 转换为高低字节
    uint8_t x_byte0 = scaled_x & 0xFF;        // X坐标低八位
    uint8_t x_byte1 = (scaled_x >> 8) & 0xFF;  // X坐标高八位
    
    uint8_t y_byte0 = scaled_y & 0xFF;        // Y坐标低八位
    uint8_t y_byte1 = (scaled_y >> 8) & 0xFF;  // Y坐标高八位
    
    uint8_t z_byte0 = scaled_z & 0xFF;        // Z坐标低八位
    uint8_t z_byte1 = (scaled_z >> 8) & 0xFF;  // Z坐标高八位
    
    // 更新数据（加锁保护）
    std::lock_guard<std::mutex> lock(data_mutex);
    x_position_data[0] = x_byte0;
    x_position_data[1] = x_byte1;
    
    y_position_data[0] = y_byte0;
    y_position_data[1] = y_byte1;
    
    z_position_data[0] = z_byte0;
    z_position_data[1] = z_byte1;
    
    ROS_INFO("接收到坐标: (%.2f, %.2f, %.2f)", x_pos, y_pos, z_pos);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "serial_sender");
    ros::NodeHandle nh;

    // 订阅目标位置话题
    ros::Subscriber target_sub = nh.subscribe("/target", 10, targetCallback);

    ROS_INFO("Serial Sender initialized.");

    ros::Publisher pub = nh.advertise<conqu_vision::ByteArray>("serial_send", 10);

    ros::Rate loop_rate(100); // 设置发送频率

    while (ros::ok()) {
        conqu_vision::ByteArray msg;

        // 获取坐标数据的副本
        std::vector<uint8_t> x_buffer, y_buffer, z_buffer;
        {
            std::lock_guard<std::mutex> lock(data_mutex);
            x_buffer = x_position_data;
            y_buffer = y_position_data;
            z_buffer = z_position_data;
        }

        // 构造完整帧
        std::vector<uint8_t> send_buffer = {
            FRAME_HEADER,       // 0xAA
            x_buffer[0], x_buffer[1],  // X坐标数据 2 字节
            y_buffer[0], y_buffer[1],  // Y坐标数据 2 字节
            z_buffer[0], z_buffer[1]   // Z坐标数据 2 字节
            // 剩余字节在后面填充
        };
        const size_t padding_size = FIXED_FRAME_SIZE - send_buffer.size() - 1; // 减去尾部占用的 1 字节
        send_buffer.insert(send_buffer.end(), padding_size, PADDING_VALUE);
        send_buffer.push_back(FRAME_FOOTER); // 添加尾部 0x66

        // 填充消息
        size_t copy_len = std::min(send_buffer.size(), msg.data.size());
        for (size_t i = 0; i < copy_len; ++i) {
            msg.data[i] = send_buffer[i];       //填充到msg中准备发送
        }
        if (copy_len < msg.data.size()) {
            std::fill(msg.data.begin() + copy_len, msg.data.end(), PADDING_VALUE);
        }

        pub.publish(msg);   //发送

        // 调试信息
        std::stringstream ss;
        ss << "Frame: ";
        for (auto b : send_buffer) {
            ss << std::hex << std::setw(2) << std::setfill('0')
               << static_cast<int>(b) << " ";
        }
        ROS_INFO("%s", ss.str().c_str());

        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}