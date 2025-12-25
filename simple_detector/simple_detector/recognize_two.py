#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
import cv2
import numpy as np
import os
import math
import threading

class RectangleDetectionNode(Node):
    def __init__(self):
        super().__init__('rectangle_detection_node_b')

        self.bridge = CvBridge()
        self.lock = threading.Lock()

        # --- 1. 参数与话题定义 (与 A 点脚本保持一致) ---
        self.declare_parameter('input_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        
        input_topic = self.get_parameter('input_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value

        # --- 2. 状态缓存 ---
        self.current_color_image = None
        self.current_depth_image = None
        self.camera_intrinsics = None
        self.has_published = False
        self.prev_rect = None

        # --- 3. 手眼标定矩阵加载 (与 A 点完全一致) ---
        calib_file_path = '/home/rpp/robot_ws/handeye_result_eye_to_hand.npz'
        try:
            calib_data = np.load(calib_file_path)
            self.R_cam2base = calib_data['R']
            self.t_cam2base = calib_data['T']
            self.get_logger().info(f'✅ 成功加载标定矩阵: {calib_file_path}')
        except Exception as e:
            self.get_logger().error(f'❌ 无法加载标定文件: {e}')
            self.R_cam2base = np.eye(3)
            self.t_cam2base = np.zeros((3, 1))

        # --- 4. ROS2 订阅器 ---
        self.color_sub = self.create_subscription(Image, input_topic, self.color_callback, 10)
        self.depth_sub = self.create_subscription(Image, depth_topic, self.depth_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, camera_info_topic, self.info_callback, 10)

        # --- 5. ROS2 发布器 ---
        self.publisher_pos_b = self.create_publisher(PoseStamped, '/vision/position_b', 10)
        self.publisher_image = self.create_publisher(Image, '/detection_image_b', 10)

        # --- 6. 定时处理循环 (5Hz) ---
        self.timer = self.create_timer(0.2, self.detection_loop)

        # 识别参数
        self.MIN_AREA = 300
        self.ASPECT_RATIO_LOW = 0.3
        self.ASPECT_RATIO_HIGH = 2.5
        self.CENTER_DIFF_THRESH = 150

        # --- 7. 显示窗口设置 ---
        self.win_name = "B-Point (Black Frame) Detection"
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)

        self.get_logger().info("RectangleDetectionNode (B点) 已启动并开启显示窗口")

    def color_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            with self.lock:
                self.current_color_image = img
        except Exception as e:
            self.get_logger().warn(f'彩色图接收失败: {e}')

    def depth_callback(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self.lock:
                self.current_depth_image = depth
        except Exception as e:
            self.get_logger().warn(f'深度图接收失败: {e}')

    def info_callback(self, msg):
        with self.lock:
            self.camera_intrinsics = {
                'fx': msg.k[0], 'fy': msg.k[4], 'cx': msg.k[2], 'cy': msg.k[5]
            }

    def pixel_to_world(self, u, v, depth_mm):
        if self.camera_intrinsics is None or depth_mm <= 0:
            return None
        z = depth_mm / 1000.0
        x = (u - self.camera_intrinsics['cx']) * z / self.camera_intrinsics['fx']
        y = (v - self.camera_intrinsics['cy']) * z / self.camera_intrinsics['fy']
        return np.array([x, y, z]).reshape(3, 1)

    def detection_loop(self):
        with self.lock:
            if self.current_color_image is None or self.current_depth_image is None:
                return
            frame = self.current_color_image.copy()
            depth_map = self.current_depth_image.copy()

        # 图像处理提取矩形
        processed_img, candidate = self.process_image(frame)

        # 逻辑：如果没有候选但有旧记录，保留旧记录以稳定显示
        if candidate is not None:
            if self.prev_rect is None or not self.is_significantly_different(self.prev_rect, candidate):
                self.prev_rect = candidate
        
        if self.prev_rect is not None:
            cx, cy = map(int, self.prev_rect['center'])
            
            # 获取深度
            h, w = depth_map.shape
            if 0 <= cx < w and 0 <= cy < h:
                roi = depth_map[max(0, cy-2):min(h, cy+3), max(0, cx-2):min(w, cx+3)]
                valid_depths = roi[roi > 0]
                depth_mm = np.median(valid_depths) if len(valid_depths) > 0 else 0

                P_cam = self.pixel_to_world(cx, cy, depth_mm)

                if P_cam is not None:
                    # 坐标变换
                    P_base = np.dot(self.R_cam2base, P_cam) + self.t_cam2base
                    xb, yb, zb = P_base.flatten()

                    # --- 绘制视觉反馈 ---
                    self._draw_rect_on_image(processed_img, self.prev_rect)
                    # 绘制中心坐标文字
                    cv2.putText(processed_img, f"B-Base:({xb:.3f}, {yb:.3f}, {zb:.3f})", (cx + 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    # 绘制十字中心
                    cv2.drawMarker(processed_img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 10, 2)

                    # --- 执行单次发布 ---
                    if not self.has_published:
                        pose_msg = PoseStamped()
                        pose_msg.header.stamp = self.get_clock().now().to_msg()
                        pose_msg.header.frame_id = "base_link"
                        pose_msg.pose.position.x = float(xb)
                        pose_msg.pose.position.y = float(yb)
                        pose_msg.pose.position.z = float(zb)
                        
                        # 固定姿态数值 (对齐 A 点脚本要求)
                        pose_msg.pose.orientation.x = 0.0
                        pose_msg.pose.orientation.y = 180.0
                        pose_msg.pose.orientation.z = 90.0
                        pose_msg.pose.orientation.w = 1.0
                        
                        self.publisher_pos_b.publish(pose_msg)
                        self.has_published = True
                        self.get_logger().info(f"🚀 B点定位成功并锁定: X:{xb:.3f} Y:{yb:.3f} Z:{zb:.3f}")

        # --- 实时窗口显示 ---
        cv2.imshow(self.win_name, processed_img)
        cv2.waitKey(1)

        # 发布监控话题图像
        try:
            self.publisher_image.publish(self.bridge.cv2_to_imgmsg(processed_img, 'bgr8'))
        except:
            pass

    def process_image(self, frame):
        # 复制一份用于绘制，不破坏原始帧
        draw_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 高斯模糊减少噪声
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # 边缘检测
        edges = cv2.Canny(blur, 50, 150)
        # 闭运算填充边缘空隙
        kernel = np.ones((5, 5), np.uint8)
        morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_candidate = None
        best_score = -1.0
        
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # 我们寻找四边形
            if len(approx) == 4:
                rect = cv2.minAreaRect(approx)
                rw, rh = rect[1]
                if rw == 0 or rh == 0: continue
                
                area = rw * rh
                aspect = max(rw, rh) / (min(rw, rh) + 1e-9)
                
                # 面积和长宽比过滤
                if area < self.MIN_AREA or not (self.ASPECT_RATIO_LOW <= aspect <= self.ASPECT_RATIO_HIGH):
                    continue
                
                # 边缘强度得分
                box = np.int32(cv2.boxPoints(rect))
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [box], -1, 255, -1)
                score = cv2.mean(edges, mask=mask)[0]
                
                if score > best_score:
                    best_score = score
                    best_candidate = {
                        'box': box.tolist(),
                        'center': rect[0],
                        'angle': rect[2],
                        'width': rw,
                        'height': rh,
                        'area': area
                    }
        return draw_frame, best_candidate

    def _draw_rect_on_image(self, image, rect_dict):
        box = np.array(rect_dict['box'], dtype=np.int32)
        # 画绿框
        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
        # 标注角度
        angle = rect_dict['angle']
        cv2.putText(image, f"Angle: {angle:.1f}", (int(rect_dict['center'][0]), int(rect_dict['center'][1]) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def is_significantly_different(self, prev, curr):
        # 如果中心点偏移超过阈值，认为是一个新的目标
        dist = np.linalg.norm(np.array(prev['center']) - np.array(curr['center']))
        return dist > self.CENTER_DIFF_THRESH

def main():
    rclpy.init()
    node = RectangleDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()