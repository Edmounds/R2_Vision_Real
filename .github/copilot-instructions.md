# Basketball Hoop Detection System - Copilot Instructions

这是一个基于ROS Noetic的实时篮球框检测系统，使用深度相机进行视觉处理。

## 规则
- **编程语言**: 使用C++编写代码 
- **回答语言**: 使用中文回答用户问题
- **代码风格**: 遵循ROS C++编码规范
- **修改限制**: 不要修改任何现有代码，只回答用户问题
- **文档格式**: 使用Markdown格式进行注释和文档编写
- **依赖管理**: 确保所有ROS依赖包都已在`package.xml`中声明
- **相机信息**: 相机已经校准，相机信息通过`camera_info`话题提供

## 核心架构

### 硬件集成
- **深度相机**: Orbbec Gemini 335，通过`OrbbecSDK_ROS1-2-main`包集成
- **激光雷达**: 通过`lslidar`相关包集成，使用TF变换定位
- **坐标系统**: 使用base_link作为根坐标系，相机和激光雷达位置通过TF发布

### 视觉处理系统
- **篮板检测方法**:
  1. `backboard_detector`: 基于ONNX模型的图像检测，使用OpenVINO推理引擎
  2. `backboard_detector_points`: 基于PCL的3D点云分析，用于检测篮板边缘
- **篮球架参数**: 标准场地参数硬编码(篮板底部高度2.285m，篮板长度1.8m)
- **数据流水线**:
  - 输入: 相机发布到`/camera/color/image_raw`和`/camera/depth_registered/points`
  - 处理: 检测节点处理流并发布结果到`/detection/result`
  - 输出: 目标点发布到`geometry_msgs/PointStamped`消息，供串行通信组件发送

### 通信接口
- **HTTP流**: 通过`camera_deliver`节点实现图像HTTP流化(端口8080)
- **串行通信**: `serial_sender`节点将检测结果转换为特定帧格式通过串口发送
  - 帧格式: 帧头(0xAA) + X坐标(2字节) + Y坐标(2字节) + Z坐标(2字节) + 填充 + 帧尾(0x55)

## 工作流程

### 编译命令
```bash
catkin_make_isolated --install --use-ninja
```

### 启动命令
- **点云检测方式**: `roslaunch conqu_vision backboard_detector_points.launch`
- **基于图像检测方式**: `roslaunch conqu_vision backboard_detector.launch`
- **相机独立启动**: `roslaunch conqu_vision camera_deliver.launch`
- **激光雷达启动**: `roslaunch conqu_vision lidar_start.launch`

### 配置管理
- **篮球架参数**: 在`src/conqu_vision/config/conqu_vison.yaml`中设置(注意文件名拼写错误)
- **启动文件参数**: 在`.launch`文件中使用`<param>`标签覆盖默认参数
- **RViz可视化**: 配置存储在`src/conqu_vision/rviz/`目录中

## 开发注意事项
- **坐标系统**: 检测到的篮板位置会通过TF变换转换到base_link坐标系下
- **点云处理**: 使用PCL库进行点云滤波、分割和特征提取
- **模型推理**: 使用OpenVINO进行ONNX模型推理，支持GPU加速
- **配置参数**: 所有核心参数均可在yaml配置文件和launch文件中调整
