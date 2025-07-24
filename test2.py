# 使用自定义的卷积核(滤波器)对输入图像进行二维卷积操作
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def find_close_points(lines, threshold=10):
    if lines is None or len(lines) < 2:
        return []
    
    result_points = {
        "LU": [w, h], # 左上角
        "RU": [0, h], # 右上角
        "LD": [w, 0], # 左下角
        "RD": [0, 0], # 右下角
    }
    
    def get_line_endpoints(line):
        """获取直线的两个端点"""
        x1, y1, x2, y2 = line[0]
        return [(x1, y1), (x2, y2)]
    
    def point_distance(p1, p2):
        """计算两点间距离"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    n_lines = len(lines)
    
    # 遍历所有直线对
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            line1 = lines[i]
            line2 = lines[j]
            
            x1, y1, x2, y2 = line1[0]
            x3, y3, x4, y4 = line2[0]
            k1 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
            k2 = (y4 - y3) / (x4 - x3) if (x4 - x3) != 0 else float('inf')
            tan_theta = abs((k1 - k2) / (1 + k1 * k2)) if (1 + k1 * k2) != 0 else float('inf')
            
            # 检查夹角是否在(60, 120)范围内
            if tan_theta > 1.732: # 对应夹角属于(60°, 120°)
                # 获取两条直线的端点
                points1 = get_line_endpoints(line1)
                points2 = get_line_endpoints(line2)
                
                # 检查是否存在距离小于threshold的端点对
                for p1 in points1:
                    for p2 in points2:
                        distance = point_distance(p1, p2)
                        if distance < threshold:
                            # 计算平均点
                            avg_point = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))

                            if avg_point[0] < w // 2: # 左侧
                                if avg_point[1] < h // 2: # 上侧
                                    if avg_point[0] + avg_point[1] < result_points["LU"][0] + result_points["LU"][1]:
                                        result_points["LU"] = avg_point
                                else: # 下侧
                                    if avg_point[0] - avg_point[1] < result_points["LD"][0] - result_points["LD"][1]:
                                        result_points["LD"] = avg_point
                            else:
                                if avg_point[1] < h // 2:
                                    if -avg_point[0] + avg_point[1] < -result_points["RU"][0] + result_points["RU"][1]:
                                        result_points["RU"] = avg_point
                                else:
                                    if -avg_point[0] - avg_point[1] < -result_points["RD"][0] - result_points["RD"][1]:
                                        result_points["RD"] = avg_point
    
    return result_points

# 您的原始代码 + 简单的重合点检测
img = cv.imread('cropped_depth.png')
img[img == 0] = 255
img = 255 - img

h, w = img.shape[:2]

# 预处理
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.equalizeHist(img)
img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

img_origin = img.copy()

# GrabCut
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
rect = (20, 20, img.shape[1]-20, img.shape[0]-20)
cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
mask2 = np.where((mask == cv.GC_FGD), 0, 1).astype('uint8')
img = img * mask2[:, :, np.newaxis]

# 边缘检测
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ksize = int(h * 0.06)
if ksize % 2 == 0:
    ksize += 1
gray = cv.GaussianBlur(gray, (ksize, ksize), 0)
canny = cv.Canny(gray, 80, 100)

ksize = int(h * 0.02)
if ksize % 2 == 0:
    ksize += 1
canny = cv.dilate(canny, cv.getStructuringElement(cv.MORPH_RECT, (ksize, ksize)))

# 霍夫直线检测
lines = cv.HoughLinesP(canny, 2, np.pi / 90, threshold=120, minLineLength=int(w * 0.3), maxLineGap=50)

# 找出重合的端点
close_points = find_close_points(lines, threshold=int(h * 0.1))

print(f"检测到 {len(lines) if lines is not None else 0} 条直线")
print(f"找到 {len(close_points)} 组重合点")

# 显示结果
con = np.zeros_like(img)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        color = tuple(np.random.randint(50, 256, size=3).tolist())
        cv.line(con, (x1, y1), (x2, y2), color, 2)

# 标记重合点
for pos, avg_point in close_points.items():
    cv.circle(img_origin, avg_point, 4, (0, 0, 255), -1)  # 红色实心圆
    cv.circle(img_origin, avg_point, 2, (255, 255, 255), 2)  # 白色边框

# 使用matplotlib显示（避免OpenCV显示问题）
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv.cvtColor(img_origin, cv.COLOR_BGR2RGB))
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(canny, cmap='gray')
plt.title('Processed Edges')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv.cvtColor(con, cv.COLOR_BGR2RGB))
plt.title(f'Lines & Close Points ({len(close_points)})')
plt.axis('off')

plt.tight_layout()
plt.show()