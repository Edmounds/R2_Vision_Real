#!/usr/bin/env python
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import numpy as np
from plyfile import PlyData, PlyElement

def callback(msg):
    rospy.loginfo("Processing point cloud with %d points", msg.width * msg.height)
    
    # 检查点云字段
    field_names = [field.name for field in msg.fields]
    rospy.loginfo("Point cloud fields: %s", field_names)
    
    # 将 PointCloud2 消息转换为点的列表
    points = []
    
    # 定义有效深度范围 (米)
    min_depth = 0.1  # 10cm
    max_depth = 10.0  # 10m
    
    valid_count = 0
    
    for point in pc2.read_points(msg, skip_nans=True, field_names=None):
        x, y, z, rgb = point
            
        valid_count += 1

        # RGB值以float形式存储，需要转换为整数再解包
        # 将float转换为32位整数
        import struct
        rgb_int = struct.unpack('I', struct.pack('f', rgb))[0]
        
        r = (rgb_int >> 16) & 0xff
        g = (rgb_int >> 8) & 0xff
        b = rgb_int & 0xff
        
        # print(f"Point: ({x:.3f}, {y:.3f}, {z:.3f}), RGB float: {rgb}, RGB int: {rgb_int:08x}, R:{r} G:{g} B:{b}")
        
        points.append((x, y, z, r, g, b))
    
    rospy.loginfo("Valid points: %d out of %d", valid_count, msg.width * msg.height)
    
    if valid_count == 0:
        rospy.logwarn("No valid points found!")
        return
    
    # 保存带颜色的点云 (使用double精度，匹配Open3D格式)
    np_points = np.array(points, dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    filename = 'output_rgb.ply'

    # 写入 ply 文件 (binary格式，匹配Open3D标准)
    el = PlyElement.describe(np_points, 'vertex')
    PlyData([el], text=False, byte_order='<').write(filename)
    rospy.loginfo("Saved %d points to %s", valid_count, filename)
    rospy.signal_shutdown("Done")

def main():
    rospy.init_node('pointcloud_saver', anonymous=True)
    rospy.Subscriber('/camera/depth_registered/points', PointCloud2, callback)
    rospy.spin()

if __name__ == '__main__':
    main()
