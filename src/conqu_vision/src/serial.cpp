#include <ros/ros.h>
#include <serial/serial.h>
#include <conqu_vision/ByteArray.h>
#include <std_msgs/String.h>
#include <sstream>
#include <iomanip>
#include <std_msgs/UInt8MultiArray.h>

void sendCallback(const conqu_vision::ByteArray::ConstPtr& msg, serial::Serial& ser) {
    try {
        size_t data_size = msg->data.size();
        ser.write((uint8_t*)msg->data.data(), data_size);

        std::ostringstream oss;
        for (size_t i = 0; i < data_size; i++) {
            oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(msg->data[i]) << " ";
        }
        std::string hex_str = oss.str();
        ROS_INFO("Sent: [%s]", hex_str.c_str());
    } catch (const serial::IOException& e) {
        ROS_ERROR("Failed to send data: %s", e.what());
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "serial_comm_cpp");
    ros::NodeHandle nh;
    ros::NodeHandle priv_nh("~");

    // 获取参数
    std::string port;
    int baudrate;
    priv_nh.param<std::string>("port", port, "/dev/ttyUSB0");
    priv_nh.param("baud", baudrate, 115200);

    // 初始化串口
    serial::Serial ser;
    try {
        ser.setPort(port);
        ser.setBaudrate(baudrate);
        serial::Timeout timeout = serial::Timeout::simpleTimeout(1000);
        ser.setTimeout(timeout);
        ser.setParity(serial::parity_none); 
        ser.open();
        ROS_INFO_STREAM("Connected to " << port << " at " << baudrate << " baud");
    } catch (const serial::IOException& e) {
        ROS_ERROR_STREAM("Failed to open serial port: " << e.what());
        return -1;
    }

    // 初始化发布者和订阅者
    ros::Publisher pub = nh.advertise<std_msgs::UInt8MultiArray>("serial_receive", 10);
    ros::Subscriber sub = nh.subscribe<conqu_vision::ByteArray>("serial_send", 10, boost::bind(sendCallback, _1, boost::ref(ser)));

    ros::Rate loop_rate(100); // 100Hz
    std::vector<uint8_t> buffer; // 数据缓冲区

    while (ros::ok()) {
        try {
            if (ser.available()) {
                // 读取串口数据
                size_t bytes_available = ser.available();
                uint8_t data[bytes_available];
                size_t bytes_read = ser.read(data, bytes_available);
                //ROS_INFO("Bytes available: %zu, Bytes read: %zu", bytes_available, bytes_read);
                

                // 处理缓冲区数据（这里假设按行或完整数据包处理）
                std::ostringstream hex_oss;
                // 发布接收到的原始数据
                std_msgs::UInt8MultiArray msg;
                msg.data.assign(data, data + bytes_read);
                for (size_t i = 0; i < bytes_read; i++) {
                    // 转换为十六进制用于日志
                    ROS_INFO("Received byte: %02x", data[i]);
                    hex_oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(data[i]) << " ";
                    
                }

                pub.publish(msg);

                // 记录日志
                ROS_INFO_STREAM("Received (" << bytes_read << " bytes): [Hex] " << hex_oss.str() );
            }
        } catch (const serial::IOException& e) {
            ROS_ERROR_STREAM("Error reading from serial port: " << e.what());
        }

        ros::spinOnce();
        loop_rate.sleep();
    }

    // 关闭串口
    try {
        ser.close();
        ROS_INFO("Serial port closed");
    } catch (const serial::IOException& e) {
        ROS_ERROR_STREAM("Failed to close serial port: " << e.what());
    }

    return 0;
}