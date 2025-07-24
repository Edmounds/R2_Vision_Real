#include "ros/ros.h"
#include "std_msgs/UInt8MultiArray.h"
#include "std_msgs/Int32MultiArray.h"
#include <vector>
#include <iostream>

// 帧头和帧尾定义
const uint8_t FRAME_HEAD = 0x98;  // 帧头为 0x98
const uint8_t FRAME_TAIL = 0x34;  // 帧尾为 0x34

ros::Publisher processed_data_pub;  // 发布者

void callback(const std_msgs::UInt8MultiArray::ConstPtr& msg) {
    const std::vector<uint8_t>& data = msg->data;

    // 查找帧头和帧尾
    auto head_it = std::find(data.begin(), data.end(), FRAME_HEAD);
    auto tail_it = std::find(data.begin(), data.end(), FRAME_TAIL);

    // 确保帧头在帧尾之前，并且帧头和帧尾都存在
    if (head_it != data.end() && tail_it != data.end() && head_it < tail_it) {
        // 提取帧头后的数据（不包括帧头和帧尾）
        std::vector<uint8_t> payload(head_it + 1, tail_it);

        // 检查数据长度是否为偶数
        if (payload.size() % 2 != 0) {
            ROS_WARN("Payload size is not even, skipping...");
            return;
        }

        // 合并每两位数据
        std_msgs::Int32MultiArray output_msg;
        for (size_t i = 0; i < payload.size(); i += 2) {
            uint8_t high = payload[i];       // 高八位
            uint8_t low = payload[i + 1];   // 低八位
            int16_t combined = (high << 8) | low;  // 合并为 16 位整数
            output_msg.data.push_back(combined);   // 存入消息
        }

        // 发布合并后的数据
        processed_data_pub.publish(output_msg);

        // 打印合并后的十进制结果
        ROS_INFO("Combined decimal values:");
        for (int32_t value : output_msg.data) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    } else {
        ROS_WARN("Frame head or tail not found, or invalid frame order.");
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "frame_processor_node");
    ros::NodeHandle nh;

    // 订阅话题
    ros::Subscriber sub = nh.subscribe("serial_receive", 10, callback);

    // 发布处理后的数据
    processed_data_pub = nh.advertise<std_msgs::Int32MultiArray>("processed_data", 10);

    ros::spin();
    return 0;
}