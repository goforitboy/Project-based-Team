import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
import cv2
import numpy as np
import pyrealsense2 as rs 
import os
import math

class RectangleDetectionNode(Node):
    def __init__(self):
        super().__init__('rectangle_detection_node')

        self.bridge = CvBridge()

        extrinsics_path = os.path.join(os.path.dirname(__file__), "extrinsics.npy")
        if os.path.exists(extrinsics_path):
            data = np.load(extrinsics_path, allow_pickle=True).item()
            self.R_extr = np.array(data["R"])
            self.T_extr = np.array(data["T"]).reshape(3, 1)
            self.get_logger().info("Loaded external calibration successfully.")
        else:
            self.R_extr = np.eye(3)
            self.T_extr = np.zeros((3, 1))
            self.get_logger().warn("⚠️ extrinsics.npy not found — using camera frame.")
        
        # 初始化 RealSense 深度相机
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # 启动彩色 + 深度流
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)  # 对齐深度到彩色图像
        self.depth_intrinsics = None  # 稍后从 profile 提取
        self.get_logger().info("RealSense D435i initialized successfully")

        # ROS2 话题发布
        self.publisher_image = self.create_publisher(Image, '/processed_image', 10)
        self.publisher_center = self.create_publisher(String, '/rectangle_center', 10)
        # 新增：发布欧拉角的主题
        self.publisher_euler = self.create_publisher(PoseStamped, '/rectangle_euler', 10)

        # 状态缓存
        self.prev_rect = None
        self.is_first_detection = True

        # 参数部分
        # 第一次识别（严格）
        self.MIN_AREA_STRICT = 500
        self.MIN_CLARITY_STRICT = 5.0
        self.ASPECT_RATIO_STRICT_LOW = 0.5
        self.ASPECT_RATIO_STRICT_HIGH = 2.0

        # 后续识别（宽松）
        self.MIN_AREA = 300
        self.MIN_CLARITY = 3.0
        self.ASPECT_RATIO_LOW = 0.3
        self.ASPECT_RATIO_HIGH = 2.5

        # 帧间比较阈值
        self.CENTER_DIFF_THRESH = 150
        self.SIZE_DIFF_RATIO = 0.6
        self.ANGLE_DIFF_THRESH = 25.0
        self.CLARITY_DROP_RATIO = 0.8

        # 实时显示窗口
        self.win_name = "Processed Image"
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)

        self.get_logger().info("RectangleDetectionNode started")

    def listener_callback(self):  # ← 新增：独立函数代替 listener_callback
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                return

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            if self.depth_intrinsics is None:
                self.depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        except Exception as e:
            self.get_logger().error(f"Frame capture failed: {e}")
            return

        processed_image, candidate = self.process_image(color_image)
        chosen = None

        ### --- 新逻辑: 始终以 prev_rect 为主 ---
        if candidate is None and self.prev_rect is not None:
            chosen = self.prev_rect  # 没检测到新矩形，保持旧框
        elif candidate is not None:
            # 第一次检测或判断是否更新
            if self.prev_rect is None:
                if self._first_detection_accept(candidate):
                    chosen = candidate
                    self.prev_rect = candidate.copy()
                    self.is_first_detection = False
                else:
                    chosen = None
            else:
                if not self.is_significantly_different(self.prev_rect, candidate):
                    chosen = candidate
                    self.prev_rect = candidate.copy()  # 接受新框
                else:
                    chosen = self.prev_rect  # 保留上次框

        # --- 始终绘制上一次的矩形（稳定显示） ---
        if self.prev_rect is not None:
            self._draw_rect_on_image(processed_image, self.prev_rect)

        # --- 获取深度并输出 3D 坐标与欧拉角 ---
        if self.prev_rect is not None and depth_frame is not None:
            cx, cy = map(int, self.prev_rect['center'])
            # 防止越界
            cx = max(0, min(cx, depth_frame.get_width() - 1))
            cy = max(0, min(cy, depth_frame.get_height() - 1))

            depth_value = depth_frame.get_distance(cx, cy)
            point_cam = np.array(rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [cx, cy], depth_value)).reshape(3, 1)

            # 将相机坐标转换到机械臂基座坐标
            point_base = self.R_extr @ point_cam + self.T_extr

            Xb, Yb, Zb = point_base.flatten()
            self.prev_rect['world'] = (Xb, Yb, Zb)
            self.prev_rect['depth'] = depth_value

            cv2.putText(processed_image,
                        f"Base: ({Xb:.3f},{Yb:.3f},{Zb:.3f})",
                        (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            self.get_logger().info(f"3D坐标: X={Xb:.3f}m, Y={Yb:.3f}m, Z={Zb:.3f}m")
           
            # --- 计算矩形在基座坐标系下的欧拉角 ---
            rect_angle_deg = float(self.prev_rect.get('angle', 0.0))
            # 将检测到的 2D 图像角度视为绕相机光轴 (Z_cam) 的旋转
            theta = math.radians(rect_angle_deg)
            Rz = np.array([
                [math.cos(theta), -math.sin(theta), 0.0],
                [math.sin(theta),  math.cos(theta), 0.0],
                [0.0,              0.0,             1.0]
            ])
            # 将矩形旋转从相机坐标系变换到基座坐标系
            R_rect_base = self.R_extr @ Rz

            # 将旋转矩阵转换为欧拉角（使用 Z-Y-X (yaw-pitch-roll) 顺序）
            def rotation_matrix_to_euler_xyz(R):
                # returns roll (x), pitch (y), yaw (z) in radians using ZYX convention
                sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
                singular = sy < 1e-6
                if not singular:
                    roll = math.atan2(R[2,1], R[2,2])
                    pitch = math.atan2(-R[2,0], sy)
                    yaw = math.atan2(R[1,0], R[0,0])
                else:
                    # Gimbal lock
                    roll = math.atan2(-R[1,2], R[1,1])
                    pitch = math.atan2(-R[2,0], sy)
                    yaw = 0.0
                return roll, pitch, yaw

            roll_rad, pitch_rad, yaw_rad = rotation_matrix_to_euler_xyz(R_rect_base)
            roll_deg = math.degrees(roll_rad)
            pitch_deg = math.degrees(pitch_rad)
            yaw_deg = math.degrees(yaw_rad)

            # 保存到 prev_rect 并在图上标注
            self.prev_rect['euler'] = (roll_deg, pitch_deg, yaw_deg)
            cv2.putText(processed_image,
                        f"Eulers: R={roll_deg:.1f} P={pitch_deg:.1f} Y={yaw_deg:.1f}",
                        (cx + 10, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # 发布中心+世界坐标（原有）
            msg = String()
            msg.data = f"{int(cx)},{int(cy)},{Xb:.3f},{Yb:.3f},{Zb:.3f}"
            self.publisher_center.publish(msg)

            w=1.0
            # 新增：发布欧拉角（roll,pitch,yaw）
            euler_msg = PoseStamped()
            euler_msg.data = f"{Xb:.3f},{Yb:.3f},{Zb:.3f},{roll_deg:.6f},{pitch_deg:.6f},{yaw_deg:.6f},{w}"
            self.publisher_euler.publish(euler_msg)

        # --- 发布图像 ---
        try:
            out_msg = self.bridge.cv2_to_imgmsg(processed_image, encoding="bgr8")
            self.publisher_image.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f"cv2_to_imgmsg failed: {e}")

        cv2.imshow(self.win_name, processed_image)
        cv2.waitKey(1)

    # ---------- 以下部分与之前相同 ----------
    def _draw_rect_on_image(self, image, rect_dict):
        try:
            box = rect_dict.get('box', None)
            center = rect_dict.get('center', None)
            if box is not None:
                cv2.drawContours(image, [np.array(box, dtype=np.int32)], 0, (0, 255, 0), 2)
            if center is not None:
                cv2.circle(image, (int(center[0]), int(center[1])), 4, (0, 0, 255), -1)
        except Exception as e:
            self.get_logger().warn(f"_draw_rect_on_image failed: {e}")

    def _first_detection_accept(self, candidate):
        area = candidate.get('area', 0)
        clarity = candidate.get('clarity', 0.0)
        width = candidate.get('width', 0)
        height = candidate.get('height', 0)
        aspect = max(width, height) / (min(width, height) + 1e-9)
        return (area >= self.MIN_AREA_STRICT and clarity >= self.MIN_CLARITY_STRICT and
                self.ASPECT_RATIO_STRICT_LOW <= aspect <= self.ASPECT_RATIO_STRICT_HIGH)

    def is_significantly_different(self, prev, curr):
        try:
            if curr is None:
                return True
            prev_center = np.array(prev['center'], dtype=float)
            curr_center = np.array(curr['center'], dtype=float)
            center_dist = np.linalg.norm(prev_center - curr_center)
            prev_w, prev_h = float(prev['width']), float(prev['height'])
            curr_w, curr_h = float(curr['width']), float(curr['height'])
            w_change = abs(curr_w - prev_w) / (prev_w + 1e-9)
            h_change = abs(curr_h - prev_h) / (prev_h + 1e-9)
            prev_angle = float(prev['angle'])
            curr_angle = float(curr['angle'])
            angle_diff = abs(((curr_angle - prev_angle ) % 180))
            if angle_diff>90:
                angle_diff=180-angle_diff
            prev_clarity = float(prev.get('clarity', 0.0))
            curr_clarity = float(curr.get('clarity', 0.0))
            clarity_drop = (prev_clarity - curr_clarity) / prev_clarity if prev_clarity > 1e-6 else 0.0
            return (center_dist > self.CENTER_DIFF_THRESH or
                    w_change > self.SIZE_DIFF_RATIO or
                    h_change > self.SIZE_DIFF_RATIO or
                    angle_diff > self.ANGLE_DIFF_THRESH or
                    clarity_drop > self.CLARITY_DROP_RATIO)
        except Exception as e:
            self.get_logger().warn(f"is_significantly_different error: {e}")
            return True

    def process_image(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray_blur, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_candidate, best_score = None, -1.0
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            if peri < 10:
                continue
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) != 4:
                continue
            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cx, cy = rect[0]
            rw, rh = rect[1]
            angle = rect[2]
            
            if rw<rh:
                angle=(90.0-angle)%180
            else:
                angle*=-1
            
            width, height = int(round(rw)), int(round(rh))
            if width == 0 or height == 0:
                continue
            area = width * height
            aspect_ratio = max(width, height) / (min(width, height) + 1e-9)
            if area < self.MIN_AREA or not (self.ASPECT_RATIO_LOW <= aspect_ratio <= self.ASPECT_RATIO_HIGH):
                continue
            mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.drawContours(mask, [box], -1, 255, -1)
            edge_strength = cv2.mean(edges, mask=mask)[0]
            if edge_strength > best_score:
                best_score = edge_strength
                best_candidate = {
                    'box': box.tolist(),
                    'center': (cx, cy),
                    'width': width,
                    'height': height,
                    'angle': angle,
                    'area': area,
                    'clarity': edge_strength
                }
        return frame, best_candidate


def main():
    rclpy.init()
    node = RectangleDetectionNode()
    try:
        while rclpy.ok():
            node.listener_callback()
    except KeyboardInterrupt:
        pass
    finally:
        node.pipeline.stop()
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
