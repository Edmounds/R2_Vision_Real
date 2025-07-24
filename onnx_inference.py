# basketball_detection_onnx_inference.py

import cv2
import os
import numpy as np
import onnxruntime as ort

def preprocess_image(image_path, target_size=(640, 640)):
    """
    预处理图片: 与detect.py保持一致的预处理方式
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    print(f"原始图片尺寸: {img.shape}")
    
    # 缩放到目标尺寸 (不进行灰度化，直接处理彩色图像)
    resized_img = cv2.resize(img, target_size)
    print(f"缩放后尺寸: {resized_img.shape}")
    
    # 转换颜色空间 BGR -> RGB (这很重要!)
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    
    # 转换为ONNX格式: (1, 3, height, width) 并归一化到 [0, 1]
    input_tensor = rgb_img.transpose(2, 0, 1)  # HWC -> CHW
    input_tensor = np.expand_dims(input_tensor, axis=0)  # 添加batch维度
    input_tensor = input_tensor.astype(np.float32) / 255.0  # 归一化
    
    print(f"ONNX输入张量形状: {input_tensor.shape}")
    
    return input_tensor, resized_img, img  # 返回ONNX输入张量、预处理后的图片和原始图片

def postprocess_detections(outputs, img_shape, confidence_threshold=0.25):
    """
    后处理ONNX模型输出 - 与detect.py保持一致的处理方式
    """
    # YOLOv11 输出格式: [batch, num_classes + 4, num_anchors]
    # 其中前4个是边界框坐标 (x_center, y_center, width, height)
    # 后面是各类别的置信度
    predictions = outputs[0][0]  # 移除批次维度
    
    # 转置以便处理: [num_anchors, num_classes + 4]
    predictions = predictions.T
    
    valid_detections = []
    
    for prediction in predictions:
        # 提取边界框坐标 (中心点格式)
        x_center, y_center, width, height = prediction[:4]
        
        # 提取类别置信度
        class_scores = prediction[4:]
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]
        
        # 过滤低置信度检测
        if confidence >= confidence_threshold:
            # 转换为左上角格式 (这里不需要恢复原始尺寸，因为我们在预处理图像上绘制)
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            valid_detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(confidence),
                'class_id': int(class_id)
            })
    
    return valid_detections

def draw_detections(img, detections, class_names=None):
    """
    在图片上绘制检测结果
    """
    result_img = img.copy()
    
    for detection in detections:
        x1, y1, x2, y2 = [int(coord) for coord in detection['bbox']]
        confidence = detection['confidence']
        class_id = detection['class_id']
        
        # 获取类别名称
        if class_names and class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"Class_{class_id}"
        
        # 绘制矩形框
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 添加标签文本
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(result_img, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(result_img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return result_img

def main():
    # 检查ONNX模型文件是否存在
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        print(f"错误: ONNX模型文件 {model_path} 不存在!")
        return
    
    # 检查测试图片是否存在
    test_image = "test.png"
    if not os.path.exists(test_image):
        print(f"错误: 测试图片 {test_image} 不存在!")
        return
    
    # 预处理图片
    print("正在预处理图片...")
    try:
        input_tensor, preprocessed_img, original_img = preprocess_image(test_image)
        # 保存预处理后的图片用于调试
        cv2.imwrite("preprocessed_test_onnx.png", preprocessed_img)
        print("预处理后的图片已保存为: preprocessed_test_onnx.png")
    except Exception as e:
        print(f"预处理图片时出错: {e}")
        return
    
    # 加载ONNX模型
    print("正在加载ONNX模型...")
    try:
        # 创建ONNX Runtime推理会话
        providers = ['CPUExecutionProvider']  # 优先使用GPU，fallback到CPU
        session = ort.InferenceSession(model_path, providers=providers)
        
        # 获取输入输出信息
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]
        
        print(f"模型输入名称: {input_name}")
        print(f"模型输出名称: {output_names}")
        print(f"使用的执行提供程序: {session.get_providers()}")
        
    except Exception as e:
        print(f"加载ONNX模型时出错: {e}")
        return
    
    # 进行推理
    print("正在进行推理...")
    try:
        # 运行推理
        outputs = session.run(output_names, {input_name: input_tensor})
        print(f"推理输出形状: {[output.shape for output in outputs]}")
        
        # 后处理检测结果
        detections = postprocess_detections(outputs, preprocessed_img.shape[:2])
        print(f"检测到 {len(detections)} 个目标")
        
        if detections:
            # 找到置信度最高的检测结果
            best_detection = max(detections, key=lambda x: x['confidence'])
            
            # 创建只包含最佳检测框的列表
            best_detections = [best_detection]
            
            # 定义类别名称（根据您的模型调整）
            class_names = ["basketball_hoop"]  # 根据实际情况修改
            
            # 绘制最佳检测结果
            result_img = draw_detections(preprocessed_img, best_detections, class_names)
            
            # 保存结果图片
            output_path = "result_onnx.png"
            cv2.imwrite(output_path, result_img)
            print(f"检测结果已保存到: {output_path}")
            
            # 打印最佳检测结果信息
            x1, y1, x2, y2 = best_detection['bbox']
            confidence = best_detection['confidence']
            class_id = best_detection['class_id']
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
            
            print(f"保留置信度最高的目标:")
            print(f"  目标: {class_name} (置信度: {confidence:.2f}, 坐标: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}])")
            
        else:
            print("未检测到任何目标")
            
    except Exception as e:
        print(f"推理过程中出错: {e}")
        return

if __name__ == '__main__':
    main()
