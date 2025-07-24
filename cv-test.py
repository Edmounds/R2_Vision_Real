import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from random import randint

# 读取图像
img = cv.imread("cropped_depth.png")
# 应用Canny边缘检测
img = cv.Canny(img, 10, 50)

# 使用霍夫变换检测直线
lines = cv.HoughLinesP(img, 2, np.pi / 90, threshold=50, minLineLength=1, maxLineGap=50)

img_show = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

print(f"图像形状: {img.shape}")  # (height, width)
height, width = img.shape[:2]

L1 = L2 = R1 = R2 = U1 = U2 = D1 = D2 = None
sLMax = sRMax = sUMax = sDMax = -1000000

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

        if length < 100:
            continue

        # 计算到各参考点的距离
        dLU = min((x1 - 0) ** 2 + (y1 - 0) ** 2, (x2 - 0) ** 2 + (y2 - 0) ** 2) ** 0.5 # 左上角
        dRU = min((x1 - width) ** 2 + (y1 - 0) ** 2, (x2 - width) ** 2 + (y2 - 0) ** 2) ** 0.5 # 右上角
        dLD = min((x1 - 0) ** 2 + (y1 - height) ** 2, (x2 - 0) ** 2 + (y2 - height) ** 2) ** 0.5 # 左下角
        dRD = min((x1 - width) ** 2 + (y1 - height) ** 2, (x2 - width) ** 2 + (y2 - height) ** 2) ** 0.5 # 右下角

        sL = length - dLU * 10 - dLD * 5
        sR = length - dRU * 2 - dRD * 2
        # sU = length - dLU * 2 - dRU * 2
        # sD = length - dRD * 2 - dLD * 2

        if sL > sLMax:
            sLMax = sL
            L1, L2 = (x1, y1), (x2, y2)
        if sR > sRMax:
            sRMax = sR
            R1, R2 = (x1, y1), (x2, y2)
        # if sU > sUMax:
        #     sUMax = sU
        #     U1, U2 = (x1, y1), (x2, y2)
        # if sD > sDMax:
        #     sDMax = sD
        #     D1, D2 = (x1, y1), (x2, y2)

# 绘制检测到的直线
if L1 is not None:
    cv.line(img_show, L1, L2, (255, 0, 0), 2)    # 左边线 - 蓝色
if R1 is not None:
    cv.line(img_show, R1, R2, (0, 255, 0), 2)    # 右边线 - 绿色  
if U1 is not None:
    cv.line(img_show, U1, U2, (0, 0, 255), 2)    # 上边线 - 红色
if D1 is not None:
    cv.line(img_show, D1, D2, (255, 255, 0), 2)  # 下边线 - 青色

# # 绘制参考点
# cv.circle(img_show, L, 5, (255, 0, 0), -1)      # 左参考点
# cv.circle(img_show, R, 5, (0, 255, 0), -1)      # 右参考点  
# cv.circle(img_show, U, 5, (0, 0, 255), -1)      # 上参考点
# cv.circle(img_show, D, 5, (255, 255, 0), -1)    # 下参考点

print(f"检测结果:")
print(f"L: {L1} -> {L2}")
print(f"R: {R1} -> {R2}") 
print(f"U: {U1} -> {U2}")
print(f"D: {D1} -> {D2}")

# 使用matplotlib显示结果（避免OpenCV显示问题）
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(img_show, cv.COLOR_BGR2RGB))
plt.title('Detected Lines')
plt.axis('off')

plt.tight_layout()
plt.show()