#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/LaserScan.h>

using namespace message_filters;

class SyncImuLidar {
public:
    SyncImuLidar() {
        // 创建发布者
        synced_imu_pub_ = nh_.advertise<sensor_msgs::Imu>("/synced_imu", 10);
        synced_scan_pub_ = nh_.advertise<sensor_msgs::LaserScan>("/synced_scan", 10);

        // 创建订阅者
        imu_sub_.subscribe(nh_, "/camera/gyro_accel/sample", 10);
        scan_sub_.subscribe(nh_, "/scan", 10);


        // 设置同步策略，队列长度为10
        sync_ = new Synchronizer<sync_policies::ApproximateTime<sensor_msgs::Imu, sensor_msgs::LaserScan>>(
            sync_policies::ApproximateTime<sensor_msgs::Imu, sensor_msgs::LaserScan>(10),
            imu_sub_, scan_sub_);
        // 设置最大时间间隔为0.1秒
        sync_->setMaxIntervalDuration(ros::Duration(0.1));
        sync_->registerCallback(boost::bind(&SyncImuLidar::callback, this, _1, _2));

        ROS_INFO("Starting synchronization of /camera/gyro_accel/sample and /scan...");
    }

    ~SyncImuLidar() {
        delete sync_;
    }

private:
    void callback(const sensor_msgs::Imu::ConstPtr& imu_msg, const sensor_msgs::LaserScan::ConstPtr& scan_msg) {
        // 发布同步后的 IMU 数据
        synced_imu_pub_.publish(*imu_msg);
        // 发布同步后的激光雷达数据
        synced_scan_pub_.publish(*scan_msg);

        // 可选：打印时间戳以验证同步
        ROS_INFO_STREAM("Synced IMU timestamp: " << imu_msg->header.stamp);
        ROS_INFO_STREAM("Synced LaserScan timestamp: " << scan_msg->header.stamp);
    }

    ros::NodeHandle nh_;
    Subscriber<sensor_msgs::Imu> imu_sub_;
    Subscriber<sensor_msgs::LaserScan> scan_sub_;
    ros::Publisher synced_imu_pub_;
    ros::Publisher synced_scan_pub_;
    Synchronizer<sync_policies::ApproximateTime<sensor_msgs::Imu, sensor_msgs::LaserScan>>* sync_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "sync_imu_lidar");
    SyncImuLidar sync_imu_lidar;
    ros::spin();
    return 0;
}