#include "conqu_vison/hoop_detector.h"  

HoopDetector::HoopDetector() : private_nh_("~")
{
    // 加载参数
    loadParameters();
    
    // 订阅点云话题
    point_cloud_sub_ = nh_.subscribe("/camera/depth/points", 1, 
                                    &HoopDetector::pointCloudCallback, this);
    
    // 发布过滤后的点云和篮框位姿
    filtered_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/filtered_cloud", 1);
    hoop_pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/hoop_pose", 1);
    visualization_pub_ = nh_.advertise<visualization_msgs::Marker>("/hoop_marker", 1);
    
    ROS_INFO("hoop detector initialized");
}

void HoopDetector::loadParameters()
{
    // 加载篮框几何参数
    nh_.param<double>("/hoop/length", hoop_length_, 1.8);
    nh_.param<double>("/hoop/width", hoop_width_, 1.05);
    nh_.param<double>("/hoop/bottom_height", hoop_bottom_height_, 2.285);
    nh_.param<double>("/hoop/top_height", hoop_top_height_, 3.335);
    nh_.param<double>("/hoop/rim_height", hoop_rim_height_, 2.43);
    
    // 加载相机参数
    nh_.param<double>("/camera/height", camera_height_, 1.02);
    
    // 加载点云处理参数
    nh_.param<double>("/point_cloud/height_filter_min", height_filter_min_, 1.165);
    nh_.param<double>("/point_cloud/height_filter_max", height_filter_max_, 2.415);
    nh_.param<double>("/point_cloud/ransac/distance_threshold", ransac_distance_threshold_, 0.03);
    nh_.param<double>("/point_cloud/ransac/length_tolerance", length_tolerance_, 0.05);
    nh_.param<double>("/point_cloud/ransac/angle_tolerance", angle_tolerance_, 5.0);
    nh_.param<double>("/point_cloud/cluster/tolerance", cluster_tolerance_, 0.02);
    nh_.param<int>("/point_cloud/cluster/min_size", min_cluster_size_, 100);
    nh_.param<int>("/point_cloud/cluster/max_size", max_cluster_size_, 25000);
    
    ROS_INFO("Parameters loaded");
    ROS_INFO("Height filter range: [%.2f, %.2f]", height_filter_min_, height_filter_max_);
}

void HoopDetector::pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
    // 将ROS消息转换为PCL点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    
    // 步骤1: 高度过滤
    pcl::PointCloud<pcl::PointXYZ>::Ptr height_filtered = filterByHeight(cloud);
    
    // 步骤2: 降噪处理
    pcl::PointCloud<pcl::PointXYZ>::Ptr denoised = removeNoise(height_filtered);
    
    // 发布过滤后的点云
    sensor_msgs::PointCloud2 filtered_cloud_msg;
    pcl::toROSMsg(*denoised, filtered_cloud_msg);
    filtered_cloud_msg.header = cloud_msg->header;
    filtered_cloud_pub_.publish(filtered_cloud_msg);
    
    // 如果点云太小，跳过后续处理
    if (denoised->size() < min_cluster_size_)
    {
        ROS_WARN("Filtered point cloud too small (%zu points), skipping detection", denoised->size());
        return;
    }
    
    // 步骤3: 提取边缘
    std::vector<pcl::ModelCoefficients> edges = detectEdges(denoised);
    
    // 步骤4和5: 几何验证和坐标计算
    geometry_msgs::PoseStamped hoop_pose;
    hoop_pose.header = cloud_msg->header;
    if (validateGeometry(edges, hoop_pose))
    {
        // 发布篮框位姿
        hoop_pose_pub_.publish(hoop_pose);
        
        // 可视化篮框
        visualizeHoop(hoop_pose);
        
        ROS_INFO("hoop detected at position: [%.2f, %.2f, %.2f]",
                 hoop_pose.pose.position.x,
                 hoop_pose.pose.position.y,
                 hoop_pose.pose.position.z);
    }
}

pcl::PointCloud<pcl::PointXYZ>::Ptr HoopDetector::filterByHeight(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
    
    // 使用直通滤波器进行高度过滤
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");  // 假设z轴是高度方向
    pass.setFilterLimits(height_filter_min_, height_filter_max_);
    pass.filter(*filtered);
    
    ROS_INFO("Height filter: %zu points -> %zu points", cloud->size(), filtered->size());
    return filtered;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr HoopDetector::removeNoise(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
    
    // 使用统计滤波去除离群点
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(50);  // 考虑50个最近邻
    sor.setStddevMulThresh(1.0);  // 标准差阈值
    sor.filter(*filtered);
    
    ROS_INFO("Noise filter: %zu points -> %zu points", cloud->size(), filtered->size());
    return filtered;
}

std::vector<pcl::ModelCoefficients> HoopDetector::detectEdges(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    std::vector<pcl::ModelCoefficients> edges;
    
    // 聚类分割点云
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);
    
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance_);
    ec.setMinClusterSize(min_cluster_size_);
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);
    
    ROS_INFO("Found %zu clusters", cluster_indices.size());
    
    // 对每个聚类进行RANSAC直线拟合
    for (const auto& indices : cluster_indices)
    {
        // 提取当前聚类
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& idx : indices.indices)
        {
            cluster->push_back((*cloud)[idx]);
        }
        
        // RANSAC直线拟合
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_LINE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(ransac_distance_threshold_);
        seg.setInputCloud(cluster);
        seg.segment(*inliers, *coefficients);
        
        if (inliers->indices.size() > min_cluster_size_ / 2)
        {
            // 计算直线长度
            pcl::PointCloud<pcl::PointXYZ>::Ptr line_points(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto& idx : inliers->indices)
            {
                line_points->push_back((*cluster)[idx]);
            }
            
            // 计算直线上的点的最大距离
            pcl::PointXYZ min_pt, max_pt;
            pcl::getMinMax3D(*line_points, min_pt, max_pt);
            double line_length = sqrt(
                pow(max_pt.x - min_pt.x, 2) +
                pow(max_pt.y - min_pt.y, 2) +
                pow(max_pt.z - min_pt.z, 2)
            );
            
            // 检查线段是否可能是篮框边缘
            bool is_horizontal = fabs(coefficients->values[5]) < 0.1; // z方向分量小
            bool is_vertical = fabs(coefficients->values[5]) > 0.9;   // z方向分量大
            
            if ((is_horizontal && fabs(line_length - hoop_length_) < length_tolerance_) ||
                (is_vertical && fabs(line_length - hoop_width_) < length_tolerance_))
            {
                edges.push_back(*coefficients);
                ROS_INFO("Found potential edge, length: %.2f, is_horizontal: %d", line_length, is_horizontal);
            }
        }
    }
    
    ROS_INFO("Detected %zu potential edges", edges.size());
    return edges;
}

bool HoopDetector::validateGeometry(const std::vector<pcl::ModelCoefficients>& edges, 
                                   geometry_msgs::PoseStamped& hoop_pose)
{
    if (edges.size() < 2)
    {
        ROS_WARN("Not enough edges detected for geometry validation");
        return false;
    }
    
    // 分类水平和垂直边缘
    std::vector<pcl::ModelCoefficients> horizontal_edges;
    std::vector<pcl::ModelCoefficients> vertical_edges;
    
    for (const auto& edge : edges)
    {
        // 方向向量
        double dx = edge.values[3];
        double dy = edge.values[4];
        double dz = edge.values[5];
        
        // 如果z方向分量很小，认为是水平边
        if (fabs(dz) < 0.1)
        {
            horizontal_edges.push_back(edge);
        }
        // 如果z方向分量很大，认为是垂直边
        else if (fabs(dz) > 0.9)
        {
            vertical_edges.push_back(edge);
        }
    }
    
    ROS_INFO("Found %zu horizontal and %zu vertical edges", 
            horizontal_edges.size(), vertical_edges.size());
    
    // 需要至少一条水平边和一条垂直边
    if (horizontal_edges.empty() || vertical_edges.empty())
    {
        ROS_WARN("Need at least one horizontal and one vertical edge");
        return false;
    }
    
    // 简单方法：使用第一条水平边和第一条垂直边来计算篮框中心
    const auto& h_edge = horizontal_edges[0];
    const auto& v_edge = vertical_edges[0];
    
    // 计算水平边的中点
    double h_x = h_edge.values[0];
    double h_y = h_edge.values[1];
    double h_z = h_edge.values[2];
    
    // 方向向量
    double h_dx = h_edge.values[3];
    double h_dy = h_edge.values[4];
    double h_dz = h_edge.values[5];
    
    // 将中心位置设置为水平边的中心点
    hoop_pose.pose.position.x = h_x;
    hoop_pose.pose.position.y = h_y;
    hoop_pose.pose.position.z = h_z;
    
    // 设置方向（以后可以改进）
    hoop_pose.pose.orientation.w = 1.0;
    hoop_pose.pose.orientation.x = 0.0;
    hoop_pose.pose.orientation.y = 0.0;
    hoop_pose.pose.orientation.z = 0.0;
    
    return true;
}

void HoopDetector::visualizeHoop(const geometry_msgs::PoseStamped& pose)
{
    visualization_msgs::Marker marker;
    marker.header = pose.header;
    marker.ns = "hoop";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::CUBE;
    marker.action = visualization_msgs::Marker::ADD;
    
    marker.pose = pose.pose;
    
    // 设置篮框尺寸
    marker.scale.x = hoop_length_; // 长
    marker.scale.y = 0.05;         // 厚度
    marker.scale.z = hoop_width_;  // 宽
    
    // 红色半透明
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    marker.color.a = 0.7;
    
    marker.lifetime = ros::Duration(0.1); // 100ms
    
    visualization_pub_.publish(marker);
}