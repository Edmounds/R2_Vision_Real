#include <ros/ros.h>
#include <conqu_vision/ByteArray.h>
#include <vector>
#include <mutex>
#include <std_msgs/Float32.h>
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
std::vector<uint8_t> position_data(2, 0x00); // 存储位置数据的 2 字节（字节根据需要更改）
std::vector<uint8_t> velocity_data(2, 0x00); // 存储速度数据的 2 字节

// 位置数据回调
void positionCallback(const std_msgs::Float32::ConstPtr& msg) {
    float position = msg->data;
    uint16_t scaled_position = static_cast<uint16_t>(position * 100);//乘100方便进行表示，获取数据

    uint8_t byte0 = scaled_position & 0xFF;     //低八位储存
    uint8_t byte1 = (scaled_position >> 8) & 0xFF;      //高八位储存

    std::lock_guard<std::mutex> lock(data_mutex);
    position_data[0] = byte0;   //进行存储
    position_data[1] = byte1;
}

// 速度数据回调（逻辑同上）
void velocityCallback(const std_msgs::Float32::ConstPtr& msg) {
    float velocity = msg->data;
    uint16_t scaled_velocity = static_cast<uint16_t>(velocity * 100);

    uint8_t byte0 = scaled_velocity & 0xFF;
    uint8_t byte1 = (scaled_velocity >> 8) & 0xFF;

    std::lock_guard<std::mutex> lock(data_mutex);
    velocity_data[0] = byte0;
    velocity_data[1] = byte1;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "serial_sender");
    ros::NodeHandle nh;

    // 订阅两个话题
    ros::Subscriber pos_sub = nh.subscribe("float_topic", 10, positionCallback);
    ros::Subscriber vel_sub = nh.subscribe("shooting_velocity", 10, velocityCallback);

    ros::Publisher pub = nh.advertise<conqu_vision::ByteArray>("serial_send", 10);

    ros::Rate loop_rate(100); // 设置发送频率

    while (ros::ok()) {
        conqu_vision::ByteArray msg;

        // 获取两个数据的副本
        std::vector<uint8_t> pos_buffer, vel_buffer;
        {
            std::lock_guard<std::mutex> lock(data_mutex);
            pos_buffer = position_data;
            vel_buffer = velocity_data;
        }

        // 构造完整帧（这里可以改长度的）
        std::vector<uint8_t> send_buffer = {
            FRAME_HEADER,       // 0x34
            pos_buffer[0], pos_buffer[1],  // 位置数据 2 字节
            vel_buffer[0], vel_buffer[1]   // 速度数据 2 字节
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