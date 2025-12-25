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

        # --- 手眼标定矩阵加载 (使用绝对路径) ---
        calib_file_path = '/home/rpp/robot_ws/handeye_result_eye_to_hand.npz'
        try:
            calib_data = np.load(calib_file_path)
            self.R_cam2base = calib_data['R']
            self.t_cam2base = calib_data['T']
            self.get_logger().info(f'✅ 成功加载手眼标定矩阵: {calib_file_path}')
        except Exception as e:
            self.get_logger().error(f'❌ 无法加载标定文件: {e}')
            self.R_cam2base = np.eye(3)
            self.t_cam2base = np.zeros((3, 1))
        # ------------------------------------

        # ROS2 订阅器
        self.color_subscription = self.create_subscription(Image, input_topic, self.color_image_callback, 10)
        self.depth_subscription = self.create_subscription(Image, depth_topic, self.depth_image_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, 10)

        # ROS2 发布器
        self.detection_publisher = self.create_publisher(PoseStamped, '/vision/position_a', 10)
        self.image_publisher = self.create_publisher(Image, '/detection_image', 10)

        # 加载 YOLO 模型
        self.model = self.load_yolov8_model(model_path)

        # 定时检测
        self.timer = self.create_timer(0.2, self.detection_loop)

        self.get_logger().info(f'Simple Detector 已启动，使用模型: {model_path}，推理设备: {self.device}')

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
                region = self.current_depth_image[
                    max(0, center_y-2):min(h, center_y+3), 
                    max(0, center_x-2):min(w, center_x+3)
                ]
                valid = region[region > 0]
                if len(valid) > 0:
                    return float(np.median(valid))
        return None

    def pixel_to_world_coordinates(self, u, v, depth):
        if self.camera_intrinsics is None or depth is None:
            return None
        depth_meters = depth / 1000.0
        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']
        X = (u - cx) * depth_meters / fx
        Y = (v - cy) * depth_meters / fy
        Z = depth_meters
        return (X, Y, Z)

    def calculate_euler_angles(self, angle_deg):
        # 内部计算仍使用弧度
        angle_rad = math.radians(angle_deg)
        r = 0.0
        p = 0.0
        y = angle_rad 
        return r, p, y

    def detect_edges_and_fit_rectangle(self, image, x1, y1, x2, y2):
        """步骤1&2: 检测边缘并拟合矩形 (完全保留原始逻辑)"""
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
            edges_canny = cv2.Canny(gray, 50, 150)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
            sobel_edges = np.uint8(sobel_magnitude > 30) * 255
            edges_combined = cv2.bitwise_or(edges_canny, sobel_edges)
            
            kernel = np.ones((5, 5), np.uint8)
            edges_combined = cv2.morphologyEx(edges_combined, cv2.MORPH_CLOSE, kernel)
            edges_combined = cv2.dilate(edges_combined, kernel, iterations=1)
            
            contours, _ = cv2.findContours(edges_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None, None, 0.0, 0.0
            
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            angle = rect[2]
            width, height = rect[1]
            
            if width < height:
                angle = angle + 90
            
            angle = angle % 180
            if angle < 0:
                angle += 180
            
            box_points = cv2.boxPoints(rect)
            box_points = np.int0(box_points)
            box_global = box_points + np.array([x1_pad, y1_pad])
            rect_area = rect[1][0] * rect[1][1]
            edge_image = cv2.cvtColor(edges_combined, cv2.COLOR_GRAY2BGR)
            
            return box_global, edge_image, angle, rect_area
            
        except Exception as e:
            self.get_logger().warn(f'边缘检测和矩形拟合失败: {str(e)}')
            return None, None, 0.0, 0.0

    def draw_detection_result(self, image, yolo_box, fitted_box, edge_image, angle, area, class_name, confidence, world_xyz):
        """绘制检测结果 (完全保留原始逻辑)"""
        result_img = image.copy()
        if edge_image is not None and fitted_box is not None:
            padding = 15
            x1, y1, x2, y2 = yolo_box
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            red_edges = np.zeros_like(edge_image)
            red_edges[:,:,2] = edge_image[:,:,0]
            roi = result_img[y1_pad:y1_pad+edge_image.shape[0], x1_pad:x1_pad+edge_image.shape[1]]
            mask = edge_image[:,:,0] > 0
            roi[mask] = cv2.addWeighted(roi, 0.7, red_edges, 0.3, 0)[mask]
            
            for i in range(4):
                cv2.line(result_img, tuple(fitted_box[i]), tuple(fitted_box[(i + 1) % 4]), (0, 255, 0), 3)
            for point in fitted_box:
                cv2.circle(result_img, tuple(point), 6, (255, 0, 0), -1)
            
            center_x = int(np.mean(fitted_box[:, 0]))
            center_y = int(np.mean(fitted_box[:, 1]))
        else:
            x1, y1, x2, y2 = yolo_box
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        
        cv2.circle(result_img, (center_x, center_y), 8, (0, 0, 255), -1)
        
        if fitted_box is not None:
            line_length = 50
            end_x = int(center_x + line_length * np.cos(np.radians(angle)))
            end_y = int(center_y + line_length * np.sin(np.radians(angle)))
            cv2.arrowedLine(result_img, (center_x, center_y), (end_x, end_y), (255, 255, 0), 2)
        
        info_text = f"{class_name} {confidence:.2f}"
        if fitted_box is not None:
            info_text += f" ∠{angle:.1f}°"
        if world_xyz is not None:
            info_text += f" XYZ({world_xyz[0]:.3f},{world_xyz[1]:.3f},{world_xyz[2]:.3f})m"
        
        text_x = int(np.min(fitted_box[:, 0])) if fitted_box is not None else yolo_box[0]
        text_y = (int(np.min(fitted_box[:, 1])) - 15) if fitted_box is not None else (yolo_box[1] - 15)
        
        text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(result_img, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0], text_y + 5), (0, 0, 0), -1)
        cv2.putText(result_img, info_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_img, (center_x, center_y)

    def detection_loop(self):
        with self.lock:
            if self.current_color_image is None: return
            frame = self.current_color_image.copy()
            depth_available = self.current_depth_image is not None
        try:
            results = self.model.predict(source=frame, conf=self.confidence_threshold, verbose=False, device=self.device, imgsz=640)
            processed, detections = self.process_detections(frame, results, depth_available)
            self.publish_results(detections, processed)
            if self.show_window:
                cv2.imshow('Simple Detector (Pillboard)', processed)
                cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f'检测循环出错: {str(e)}')

    def process_detections(self, image, results, depth_available):
        result_img = image.copy()
        detections = []
        for result in results:
            if result.boxes is None: continue
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.confidence_threshold: continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cls = int(box.cls[0])
                class_name = self.model.names.get(cls, f"id_{cls}")
                
                fitted_box, edge_img, angle, area = self.detect_edges_and_fit_rectangle(image, x1, y1, x2, y2)
                c_x, c_y = (int(np.mean(fitted_box[:, 0])), int(np.mean(fitted_box[:, 1]))) if fitted_box is not None else ((x1+x2)//2, (y1+y2)//2)
                depth_v = self.get_distance(c_x, c_y) if depth_available else None
                world_xyz = self.pixel_to_world_coordinates(c_x, c_y, depth_v)
                r, p, y = self.calculate_euler_angles(angle) if fitted_box is not None else (0.0, 0.0, 0.0)
                
                result_img, center = self.draw_detection_result(result_img, (x1, y1, x2, y2), fitted_box, edge_img, angle, area, class_name, conf, world_xyz)
                detections.append({'world_xyz': world_xyz, 'euler_r': r, 'euler_p': p, 'euler_y': y, 'has_fitted_box': fitted_box is not None})
        return result_img, detections
        
    def publish_results(self, detections, processed_image):
        if detections:
            for d in detections:
                if d['world_xyz'] is not None:
                    # 1. 坐标转换 (相机 -> 基座)
                    P_cam = np.array([d['world_xyz'][0], d['world_xyz'][1], d['world_xyz'][2]]).reshape(3, 1)
                    P_base = np.dot(self.R_cam2base, P_cam) + self.t_cam2base
                    
                    # 2. 旋转角度转换 (相机 -> 基座)
                    yaw_cam = d['euler_y'] # 弧度
                    R_obj_cam = np.array([[math.cos(yaw_cam), -math.sin(yaw_cam), 0],
                                          [math.sin(yaw_cam),  math.cos(yaw_cam), 0],
                                          [0, 0, 1]])
                    R_obj_base = np.dot(self.R_cam2base, R_obj_cam)
                    new_euler_y_rad = math.atan2(R_obj_base[1, 0], R_obj_base[0, 0])
                    
                    # --- 【关键修改】: 将 z 的值转换为角度制 ---
                    new_euler_y_deg = math.degrees(new_euler_y_rad)
                    # ---------------------------------------

                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = self.get_clock().now().to_msg()
                    pose_msg.header.frame_id = "base_link"
                    pose_msg.pose.position.x = float(P_base[0])
                    pose_msg.pose.position.y = float(P_base[1])
                    pose_msg.pose.position.z = float(P_base[2])
                    
                    pose_msg.pose.orientation.x = d['euler_r']
                    pose_msg.pose.orientation.y = d['euler_p']
                    pose_msg.pose.orientation.z = new_euler_y_deg # 此时是角度值
                    pose_msg.pose.orientation.w = 0.0
                    
                    self.detection_publisher.publish(pose_msg)
                    break 
        self.image_publisher.publish(self.bridge.cv2_to_imgmsg(processed_image, 'bgr8'))

    def destroy_node(self):
        if self.show_window: cv2.destroyAllWindows()
        super().destroy_node()

def main():
    rclpy.init()
    node = SimpleDetector()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()