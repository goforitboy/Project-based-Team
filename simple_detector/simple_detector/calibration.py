#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Vector3 # 需要导入 Vector3
from geometry_msgs.msg import Pose # 还需要导入 Pose
from rclpy.executors import ExternalShutdownException
from cv_bridge import CvBridge
import pyrealsense2 as rs
import cv2
import numpy as np
import threading
import time
import os
import sys
import math

# ------------------------- 参数设置 -------------------------
CHECKERBOARD = (8, 5)   # 棋盘格内角点数量 (w, h)
SQUARE_SIZE = 0.025     # 棋盘格格子的实际边长 (米)
NUM_IMAGES = 15         # 采集图像数量
OUTPUT_FILE = "handeye_result_eye_to_hand.npy" # 区分文件名
# ------------------------------------------------------------

def beep():
    """播放提示音"""
    try:
        sys.stdout.write('\a')
        sys.stdout.flush()
    except Exception:
        pass

def is_pose_similar(pose1, pose2, pos_thresh=0.005, rot_thresh=0.05):
    """检查两个位姿是否过于接近（避免重复数据）"""
    if pose1 is None or pose2 is None:
        return False
    pos1 = pose1[1] # Translation
    pos2 = pose2[1]
    dist = np.linalg.norm(pos1 - pos2)
    
    # 简单的旋转差异检查 (这里简化处理，只看平移通常足够防止连按，严谨可加旋转)
    return dist < pos_thresh

class EyeToHandCalibration(Node):
    def __init__(self):
        super().__init__('eye_to_hand_calibration')
        self.lock = threading.Lock()

        # 1. 相机初始化
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.profile = self.pipeline.start(self.config)
            self.align = rs.align(rs.stream.color)
            self.get_logger().info("✅ Realsense D435i 启动成功")
            
            # 2. 提前获取内参 (Intrinsics)
            stream_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = stream_profile.get_intrinsics()
            self.K = np.array([[intr.fx, 0, intr.ppx],
                               [0, intr.fy, intr.ppy],
                               [0, 0, 1]], dtype=np.float64)
            self.dist_coeffs = np.array(intr.coeffs, dtype=np.float64)
            self.get_logger().info(f"📷 相机内参加载:\n{self.K}")

        except Exception as e:
            self.get_logger().error(f"❌ 相机启动失败: {e}")
            sys.exit(1)

        # 3. 订阅机械臂位姿
        self.current_ee_pos_msg = None
        self.current_ee_rpy_msg = None
        # 请确认话题名称是否正确
        self.create_subscription(PoseStamped, "/arm/end_effector_pose", self.position_callback, 10)
        self.create_subscription(Vector3, "/global/position_b", self.rpy_callback, 10)
        
        # 4. 棋盘格世界坐标 (Object Points)
        self.objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        self.objp *= SQUARE_SIZE

        # 数据容器
        self.R_gripper2base = []
        self.t_gripper2base = []
        self.R_target2cam = []
        self.t_target2cam = []
        
        self.last_collected_pose = None # 用于去重
        self.collected_count = 0
        self.running = True

        # 启动处理线程
        self.thread = threading.Thread(target=self.capture_loop)
        self.thread.start()

    def position_callback(self, msg):
        """处理位置信息，并将其存储在 self.current_ee_pos_msg 中"""
        with self.lock:
            # 只存储位置信息
            self.current_ee_pos_msg = msg.pose.position

    def rpy_callback(self, msg):
        """处理欧拉角信息 (假设 msg.x/y/z 是 Roll/Pitch/Yaw 的弧度值)"""
        with self.lock:
            # 存储 RPY 信息
            self.current_ee_rpy_msg = msg


    def quaternion_to_matrix(self, q):
        """四元数转旋转矩阵 [x, y, z, w] -> 3x3 matrix"""
        # 注意：ROS msg 的四元数通常是 (x, y, z, w)
        x, y, z, w = q.x, q.y, q.z, q.w
        # 使用 scipy 或者手动计算，这里保持你的手动计算逻辑，确保公式正确
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ], dtype=np.float64)
        return R

    def euler_to_quaternion(self, roll, pitch, yaw):
        """
        将欧拉角转换为四元数 [x, y, z, w] (旋转顺序: xyz)
        输入: roll, pitch, yaw (弧度)
        """
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return qx, qy, qz, qw

    def capture_loop(self):
        print("\n================================================")
        print("👁️  Eye-to-Hand 手眼标定程序 (眼在手外)")
        print("------------------------------------------------")
        print("📝 操作说明：")
        print("1. 【重要】将相机固定在环境中（不动）。")
        print("2. 【重要】将棋盘格固定在机械臂末端（由机械臂夹持）。")
        print("3. 移动机械臂，使棋盘格出现在相机视野的不同位置和角度。")
        print("   (注意：需要包含旋转变化，不仅是平移)")
        print("4. 当识别到棋盘格时，按 'S' 键保存当前帧。")
        print("5. 采集满 %d 张后自动计算，或按 'Q' 退出。" % NUM_IMAGES)
        print("================================================\n")

        while rclpy.ok() and self.running:
            # 获取图像
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            color_frame = aligned.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1. 检测角点
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

            # UI 显示
            display_img = frame.copy()
            
            if ret:
                # 2. 关键优化：亚像素角点细化 (Sub-pixel refinement)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                cv2.drawChessboardCorners(display_img, CHECKERBOARD, corners2, ret)
                cv2.putText(display_img, "Ready to Capture (Press S)", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(display_img, "Looking for Chessboard...", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # 显示进度
            cv2.putText(display_img, f"Count: {self.collected_count}/{NUM_IMAGES}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.imshow("Eye-to-Hand Calibration", display_img)
            key = cv2.waitKey(1) & 0xFF

            # 处理按键
            if key == ord('s') and ret:
                self.save_frame(corners2) # 传入优化后的角点
            elif key == ord('q'):
                self.running = False
                break

            if self.collected_count >= NUM_IMAGES:
                print("\n[INFO] 采集完成，开始计算...")
                self.solve_handeye()
                self.running = False
                break

        self.cleanup()

    def save_frame(self, corners):
        with self.lock:
            if self.current_ee_pos_msg is None or self.current_ee_rpy_msg is None:
                print("[WARN] ❌ 缺少机械臂位置或欧拉角数据！")
                return

            # 1. 提取位置
            pos_msg = self.current_ee_pos_msg
            pos = np.array([pos_msg.x, pos_msg.y, pos_msg.z])
            
            # 2. 提取 RPY
            rpy_msg = self.current_ee_rpy_msg
            roll_rad = rpy_msg.x
            pitch_rad = rpy_msg.y
            yaw_rad = rpy_msg.z

        # 检查是否与上一帧重复 (使用位置去重)
        current_data = (None, pos)
        if self.last_collected_pose and is_pose_similar(self.last_collected_pose, current_data):
            print("[WARN] ⚠️ 位姿移动太小，请移动机械臂后再采集。")
            return

        # 3. 欧拉角 -> 四元数 (手动转换)
        qx, qy, qz, qw = self.euler_to_quaternion(roll_rad, pitch_rad, yaw_rad)
        
        # 4. 构造一个临时的四元数对象，以便调用现有的 quaternion_to_matrix
        class TempQuat:
            def __init__(self, x, y, z, w): self.x, self.y, self.z, self.w = x, y, z, w
        temp_q = TempQuat(qx, qy, qz, qw)

        # 5. 四元数 -> 旋转矩阵 (使用现有的方法)
        R_robot = self.quaternion_to_matrix(temp_q)
        t_robot = pos.reshape(3, 1)
        
        # ... (后续计算和存储逻辑保持不变) ...
        # self.R_gripper2base.append(R_robot)
        # self.t_gripper2base.append(t_robot)
        # ...

    def solve_handeye(self):
        if self.collected_count < 3:
            print("[ERROR] 数据不足，无法标定")
            return

        print("正在进行 Eye-to-Hand 标定计算 (Method: TSAI)...")
        
        try:
            # Eye-to-Hand: 求解 相机 在 机器人基座 坐标系下的位姿
            # 输入: 
            # 1. Gripper相对于Base的位姿 (Robot FK)
            # 2. Target相对于Camera的位姿 (SolvePnP)
            R_cam2base, t_cam2base = cv2.calibrateHandEye(
                self.R_gripper2base,
                self.t_gripper2base,
                self.R_target2cam,
                self.t_target2cam,
                method=cv2.CALIB_HAND_EYE_TSAI
            )

            print("\n" + "="*40)
            print("       🏆 标定结果 (Camera in Base)       ")
            print("="*40)
            print("旋转矩阵 R:\n", R_cam2base)
            print("平移向量 t (meters):\n", t_cam2base)
            print("-" * 40)
            
            # 验证建议
            print("💡 验证方法：")
            print("将结果填入机械臂或视觉程序中，控制机械臂末端移动到棋盘格中心，")
            print("看实际物理位置是否重合。")

            np.savez(OUTPUT_FILE.replace('.npy', '.npz'), R=R_cam2base, T=t_cam2base)
            print(f"✅ 结果已保存至 {OUTPUT_FILE.replace('.npy', '.npz')}")
        
        except cv2.error as e:
            print(f"[ERROR] OpenCV 计算错误: {e}")

    def cleanup(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()
        # 关闭ROS节点
        self.destroy_node()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = EyeToHandCalibration()
    # 使用 spin 保持 ROS 通讯活跃，但具体的 loop 在线程中运行
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        pass

if __name__ == "__main__":
    main()