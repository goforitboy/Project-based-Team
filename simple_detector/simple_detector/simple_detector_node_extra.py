#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
import cv2
import numpy as np
from ultralytics import YOLO
import os
import threading
import math

class SimpleDetector(Node):
    def __init__(self):
        super().__init__('simple_detector')

        # 参数定义
        self.declare_parameter('input_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('confidence_threshold', 0.90)
        self.declare_parameter('show_window', True)
        
        # 新增参数: 显式指定推理设备 ('cpu' 或 '0'/'cuda:0' for GPU)
        self.declare_parameter('device', 'cpu') # 默认使用 CPU
        
        package_dir = os.path.dirname(os.path.abspath(__file__))
        default_model_path = os.path.join(package_dir, 'models', 'pillboard_with_negatives_best.pt')
        self.declare_parameter('model_path', default_model_path)

        input_topic = self.get_parameter('input_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.show_window = self.get_parameter('show_window').value
        model_path = self.get_parameter('model_path').value
        self.device = self.get_parameter('device').value # 获取设备参数

        self.bridge = CvBridge()
        self.current_color_image = None
        self.current_depth_image = None
        self.camera_intrinsics = None
        self.lock = threading.Lock()

        # ROS2 订阅器
        self.color_subscription = self.create_subscription(Image, input_topic, self.color_image_callback, 10)
        self.depth_subscription = self.create_subscription(Image, depth_topic, self.depth_image_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, 10)

        # ROS2 发布器 - 只修改话题名称
        self.detection_publisher = self.create_publisher(PoseStamped, '/vision/position_a', 10)
        self.image_publisher = self.create_publisher(Image, '/detection_image', 10)

        # 加载 YOLO 模型
        self.model = self.load_yolov8_model(model_path)

        # 定时检测：【优化 1】从 0.05s 降低到 0.2s (从 20 FPS 降低到 5 FPS)
        self.timer = self.create_timer(0.2, self.detection_loop)

        self.get_logger().info(f'Simple Detector 已启动，使用模型: {model_path}，推理设备: {self.device}')

    # ... load_yolov8_model, color_image_callback, depth_image_callback, camera_info_callback, 
    #     get_distance, pixel_to_world_coordinates, calculate_euler_angles, 
    #     detect_edges_and_fit_rectangle, draw_detection_result 保持不变 ...
    
    def load_yolov8_model(self, model_path):
        """加载 YOLOv8 模型"""
        self.get_logger().info('正在加载 YOLOv8 模型...')
        abs_path = os.path.abspath(model_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"模型文件不存在: {abs_path}")
        model = YOLO(abs_path)
        self.get_logger().info(f'YOLOv8 模型加载成功: {abs_path}')
        return model

    def color_image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            with self.lock:
                self.current_color_image = cv_image
        except Exception as e:
            self.get_logger().warn(f'彩色图像处理失败: {str(e)}')

    def depth_image_callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self.lock:
                self.current_depth_image = depth_image
        except Exception as e:
            self.get_logger().warn(f'深度图像处理失败: {str(e)}')

    def camera_info_callback(self, msg):
        """接收相机内参"""
        self.camera_intrinsics = {
            'fx': msg.k[0],
            'fy': msg.k[4],
            'cx': msg.k[2],
            'cy': msg.k[5],
            'width': msg.width,
            'height': msg.height
        }
        self.get_logger().info('已接收到相机内参')

    def get_distance(self, center_x, center_y):
        """获取指定像素点的深度距离（毫米）"""
        with self.lock:
            if self.current_depth_image is None:
                return None
            h, w = self.current_depth_image.shape
            if 0 <= center_x < w and 0 <= center_y < h:
                # 取5x5区域的中值，减少噪声影响
                region = self.current_depth_image[
                    max(0, center_y-2):min(h, center_y+3), 
                    max(0, center_x-2):min(w, center_x+3)
                ]
                valid = region[region > 0]
                if len(valid) > 0:
                    return float(np.median(valid))
        return None

    def pixel_to_world_coordinates(self, u, v, depth):
        """
        将像素坐标(u,v)和深度值转换为三维世界坐标(X,Y,Z)
        坐标系定义：
        X: 相机右侧为正（米）
        Y: 相机下方为正（米） 
        Z: 相机前方为正（米）
        """
        if self.camera_intrinsics is None or depth is None:
            return None
        
        # 深度值单位转换（毫米转米）
        depth_meters = depth / 1000.0
        
        # 相机内参
        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']
        
        # 像素坐标系到相机坐标系的转换
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        # Z = depth
        
        X = (u - cx) * depth_meters / fx
        Y = (v - cy) * depth_meters / fy
        Z = depth_meters
        
        return (X, Y, Z)

    def calculate_euler_angles(self, angle_deg):
        """
        计算欧拉角r, p, y
        基于矩形框的角度计算欧拉角
        """
        # 将角度转换为弧度
        angle_rad = math.radians(angle_deg)
        
        # 假设绕Z轴的旋转角就是检测到的角度
        # 绕X轴和Y轴的旋转角设为0（假设目标在平面上）
        r = 0.0  # 绕X轴的旋转角（横滚角）
        p = 0.0  # 绕Y轴的旋转角（俯仰角）
        y = angle_rad  # 绕Z轴的旋转角（偏航角）
        
        return r, p, y

    def detect_edges_and_fit_rectangle(self, image, x1, y1, x2, y2):
        """步骤1&2: 检测边缘并拟合矩形"""
        try:
            # 步骤1: 提取检测区域
            padding = 15
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(image.shape[1], x2 + padding)
            y2_pad = min(image.shape[0], y2 + padding)
            
            roi = image[y1_pad:y2_pad, x1_pad:x2_pad]
            if roi.size == 0:
                return None, None, 0.0, 0.0
            
            # 步骤2: 边缘检测
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # 多种边缘检测方法组合
            edges_canny = cv2.Canny(gray, 50, 150)
            
            # Sobel边缘
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
            sobel_edges = np.uint8(sobel_magnitude > 30) * 255
            
            # 合并边缘
            edges_combined = cv2.bitwise_or(edges_canny, sobel_edges)
            
            # 形态学操作强化边缘
            kernel = np.ones((5, 5), np.uint8)
            edges_combined = cv2.morphologyEx(edges_combined, cv2.MORPH_CLOSE, kernel)
            edges_combined = cv2.dilate(edges_combined, kernel, iterations=1)
            
            # 查找轮廓
            contours, _ = cv2.findContours(edges_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None, None, 0.0, 0.0
            
            # 找到最大的轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 步骤3: 拟合最小外接矩形
            rect = cv2.minAreaRect(largest_contour)
            angle = rect[2]
            
            # 修改角度计算：矩形框长边与x轴正方向的角度，y轴向上
            width, height = rect[1]
            
            # 确保角度在0-180度范围内
            if width < height:
                angle = angle + 90
            
            # 规范化角度到0-180度
            angle = angle % 180
            if angle < 0:
                angle += 180
            
            # 获取矩形四个点
            box_points = cv2.boxPoints(rect)
            box_points = np.int0(box_points)
            
            # 转换到原图坐标系
            box_global = box_points + np.array([x1_pad, y1_pad])
            
            # 计算面积
            rect_area = rect[1][0] * rect[1][1]
            
            # 返回边缘图像和拟合的矩形
            edge_image = cv2.cvtColor(edges_combined, cv2.COLOR_GRAY2BGR)
            
            return box_global, edge_image, angle, rect_area
            
        except Exception as e:
            self.get_logger().warn(f'边缘检测和矩形拟合失败: {str(e)}')
            return None, None, 0.0, 0.0

    def draw_detection_result(self, image, yolo_box, fitted_box, edge_image, angle, area, class_name, confidence, world_xyz):
        """绘制检测结果：显示边缘和拟合的矩形"""
        result_img = image.copy()
        
        # 步骤4: 在原始图像上绘制边缘（红色）
        if edge_image is not None and fitted_box is not None:
            # 将边缘叠加到原图（红色显示）
            padding = 15
            x1, y1, x2, y2 = yolo_box
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            
            # 创建红色边缘图像
            red_edges = np.zeros_like(edge_image)
            red_edges[:,:,2] = edge_image[:,:,0]  # 红色通道
            
            # 将红色边缘叠加到原图
            roi = result_img[y1_pad:y1_pad+edge_image.shape[0], x1_pad:x1_pad+edge_image.shape[1]]
            mask = edge_image[:,:,0] > 0
            roi[mask] = cv2.addWeighted(roi, 0.7, red_edges, 0.3, 0)[mask]
            
            # 步骤5: 绘制拟合的矩形框（绿色）
            for i in range(4):
                cv2.line(result_img, tuple(fitted_box[i]), tuple(fitted_box[(i + 1) % 4]), 
                        (0, 255, 0), 3)
            
            # 绘制矩形角点
            for point in fitted_box:
                cv2.circle(result_img, tuple(point), 6, (255, 0, 0), -1)
            
            # 计算中心点
            center_x = int(np.mean(fitted_box[:, 0]))
            center_y = int(np.mean(fitted_box[:, 1]))
            
        else:
            # 如果边缘检测失败，使用原始YOLO框
            x1, y1, x2, y2 = yolo_box
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        
        # 绘制中心点
        cv2.circle(result_img, (center_x, center_y), 8, (0, 0, 255), -1)
        
        # 绘制方向线
        if fitted_box is not None:
            line_length = 50
            end_x = int(center_x + line_length * np.cos(np.radians(angle)))
            end_y = int(center_y + line_length * np.sin(np.radians(angle)))
            cv2.arrowedLine(result_img, (center_x, center_y), (end_x, end_y), (255, 255, 0), 2)
        
        # 添加信息标签（包含完整的三维坐标）
        info_text = f"{class_name} {confidence:.2f}"
        if fitted_box is not None:
            info_text += f" ∠{angle:.1f}°"
        
        if world_xyz is not None:
            info_text += f" XYZ({world_xyz[0]:.3f},{world_xyz[1]:.3f},{world_xyz[2]:.3f})m"
        
        # 确定标签位置
        if fitted_box is not None:
            text_x = int(np.min(fitted_box[:, 0]))
            text_y = int(np.min(fitted_box[:, 1])) - 15
        else:
            text_x = yolo_box[0]
            text_y = yolo_box[1] - 15
        
        if text_y < 10:
            text_y = (yolo_box[3] + 30) if fitted_box is None else (int(np.max(fitted_box[:, 1])) + 30)
        
        # 绘制标签背景
        text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(result_img, 
                     (text_x, text_y - text_size[1] - 5), 
                     (text_x + text_size[0], text_y + 5), 
                     (0, 0, 0), -1)
        
        # 绘制标签文字
        cv2.putText(result_img, info_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_img, (center_x, center_y)

    def detection_loop(self):
        with self.lock:
            if self.current_color_image is None:
                return
            frame = self.current_color_image.copy()
            depth_available = self.current_depth_image is not None

        try:
            # 【优化 2】显式指定设备和图像尺寸（如 640）来控制推理负载
            results = self.model.predict(
                source=frame, 
                conf=self.confidence_threshold, 
                verbose=False,
                device=self.device,
                imgsz=640 # 根据您的模型训练尺寸设置，以减少自适应开销
            )
            
            processed, detections = self.process_detections(frame, results, depth_available)
            self.publish_results(detections, processed)

            if self.show_window:
                cv2.imshow('Simple Detector (Pillboard)', processed)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.destroy_node()
                    rclpy.shutdown()
        except Exception as e:
            self.get_logger().error(f'检测循环出错: {str(e)}')

    def process_detections(self, image, results, depth_available):
        result_img = image.copy()
        detections = []

        for result in results:
            if result.boxes is None:
                continue
                
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue
                    
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cls = int(box.cls[0])
                class_name = self.model.names.get(cls, f"id_{cls}")
                
                # 步骤1&2: 检测边缘并拟合矩形 - 保持不变
                fitted_box, edge_image, angle, area = self.detect_edges_and_fit_rectangle(image, x1, y1, x2, y2)
                
                # 计算中心点
                if fitted_box is not None:
                    center_x = int(np.mean(fitted_box[:, 0]))
                    center_y = int(np.mean(fitted_box[:, 1]))
                else:
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                
                # 获取深度距离并转换为三维世界坐标
                depth_value = self.get_distance(center_x, center_y) if depth_available else None
                world_xyz = self.pixel_to_world_coordinates(center_x, center_y, depth_value) if depth_value else None
                
                # 计算欧拉角
                euler_r, euler_p, euler_y = self.calculate_euler_angles(angle) if fitted_box is not None else (0.0, 0.0, 0.0)
                
                # 步骤4&5: 绘制结果
                result_img, center = self.draw_detection_result(
                    result_img, (x1, y1, x2, y2), fitted_box, edge_image, angle, area, 
                    class_name, conf, world_xyz
                )
                
                detections.append({
                    'class_name': class_name,
                    'confidence': conf,
                    'center': center,
                    'distance': depth_value,
                    'world_xyz': world_xyz,
                    'rotation_angle': angle,
                    'euler_r': euler_r,
                    'euler_p': euler_p,
                    'euler_y': euler_y,
                    'area': area,
                    'has_fitted_box': fitted_box is not None
                })

        # 显示统计信息
        fitted_count = sum(1 for d in detections if d['has_fitted_box'])
        cv2.putText(result_img, f"Detections: {len(detections)} Fitted: {fitted_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return result_img, detections
        
    def publish_results(self, detections, processed_image):
        if detections:
            for d in detections:
                if d['world_xyz'] is not None:
                    # 创建 PoseStamped 消息
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = self.get_clock().now().to_msg()
                    pose_msg.header.frame_id = "camera_frame"
                    
                    # 设置位置坐标
                    pose_msg.pose.position.x = d['world_xyz'][0]
                    pose_msg.pose.position.y = d['world_xyz'][1]
                    pose_msg.pose.position.z = d['world_xyz'][2]
                    
                    # 设置欧拉角
                    pose_msg.pose.orientation.x = d['euler_r']
                    pose_msg.pose.orientation.y = d['euler_p'] 
                    pose_msg.pose.orientation.z = d['euler_y']
                    pose_msg.pose.orientation.w = 0.0
                    
                    # 发布 PoseStamped 消息
                    self.detection_publisher.publish(pose_msg)
                    break  # 只发布第一个有效目标
        # 保持图像发布不变
        img_msg = self.bridge.cv2_to_imgmsg(processed_image, 'bgr8')
        self.image_publisher.publish(img_msg)

    def destroy_node(self):
        if self.show_window:
            cv2.destroyAllWindows()
        super().destroy_node()

def main():
    rclpy.init()
    node = SimpleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()