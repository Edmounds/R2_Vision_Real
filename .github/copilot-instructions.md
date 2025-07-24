# Basketball Hoop Detection System - Copilot Instructions

This is a ROS Noetic-based computer vision system for real-time basketball hoop detection using depth cameras.

## Rules for Copilot
- **Language**: Use C++ 
- **Language**: Use Chinese for answers
- **Code Style**: Follow ROS C++ conventions
- **Rules**:Don't edit any of my codes,only output answers to my questions
- **Documentation**: Use Markdown format for comments and documentation
- **Dependencies**: Ensure all required ROS packages are included in `package.xml`
- **Camera_information**: The camera has already been calibrated and the camera information is provided in the `camera_info` topic.

### Core Components
- **Hardware Integration**: Orbbec Gemini 335 depth camera via `OrbbecSDK_ROS1-2-main` package
- **Vision Processing**: Two detection approaches in `conqu_vision` package:
  - `backboard_detector`: ONNX model-based image detection using OpenVINO
  - `backboard_detector_points`: PCL-based 3D point cloud analysis for backboard edge detection
- **Basketball Parameters**: Hardcoded for standard courts (2.285m height, 1.8m backboard length)


### Data Flow
1. Orbbec camera publishes to `/camera/color/image_raw` and `/camera/depth_registered/points`
2. Detection nodes process streams and publish results to `/detection/result`  
3. RViz visualization shows detection overlays and 3D point clouds
4. TF transforms link camera poses to base_link coordinate frame

## Build & Development Workflow


### Launch Patterns
- **Point cloud detection**: `roslaunch conqu_vision backboard_detector_points.launch`
- **Image-based detection**: `roslaunch conqu_vision backboard_detector.launch`
- Camera streams auto-launch via `orbbec_camera` include in launch files


### Configuration Management
- Camera parameters: `src/conqu_vision/config/conqu_vison.yaml` (note typo in filename)
- Launch-time parameter overrides in `.launch` files using `<param>` tags
- RViz configs stored in `src/conqu_vision/rviz/` for visualization presets


