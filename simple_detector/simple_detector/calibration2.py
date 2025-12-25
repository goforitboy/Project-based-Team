#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
import cv2
import numpy as np
import threading
import sys
import math

# ------------------------- 参数设置 -------------------------
SQUARE_SIZE = 0.024      # 🔴 再次确认：单位是米！(比如2.5cm写0.025)
CHECKERBOARD = (8, 5)    # 棋盘格内角点数
NUM_IMAGES = 4          
OUTPUT_FILE = "handeye_result_eye_to_hand.npy" 
# ------------------------------------------------------------

class EyeToHandCalibration(Node):
    def __init__(self):
        super().__init__('eye_to_hand_calibration')
        self.lock = threading.Lock()

        # 相机初始化
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.profile = self.pipeline.start(self.config)
            self.align = rs.align(rs.stream.color)
            
            stream_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = stream_profile.get_intrinsics()
            self.K = np.array([[intr.fx, 0, intr.ppx],
                               [0, intr.fy, intr.ppy],
                               [0, 0, 1]], dtype=np.float64)
            self.dist_coeffs = np.array(intr.coeffs, dtype=np.float64)
        except Exception as e:
            print(f"❌ 相机启动失败: {e}")
            sys.exit(1)

        self.objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        self.objp *= SQUARE_SIZE

        # 注意：这里的列表将存储 "Gripper to Base" (反向) 的位姿
        self.R_gripper2base = []
        self.t_gripper2base = []
        self.R_target2cam = []
        self.t_target2cam = []
        
        self.collected_count = 0
        self.running = True
        self.thread = threading.Thread(target=self.capture_loop)
        self.thread.start()

    def euler_to_matrix(self, roll, pitch, yaw):
        """欧拉角(sXYZ) -> 旋转矩阵"""
        Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def capture_loop(self):
        print("\n================================================")
        print("    🔧 最终修正版 (自动矩阵求逆)")
        print("================================================")
        print("⚠️  输入提示：请继续输入 tf2_echo 看到的原始数据 (Base->Tool)")
        print("    程序会自动帮你取反，不需要你手动算！")
        
        while rclpy.ok() and self.running:
            frames = self.pipeline.wait_for_frames()
            color_frame = self.align.process(frames).get_color_frame()
            if not color_frame: continue

            frame = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
            
            display_img = frame.copy()
            corners2 = None
            
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(display_img, CHECKERBOARD, corners2, ret)
                cv2.putText(display_img, "Ready! Press 'S'", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(display_img, "No Board", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(display_img, f"Count: {self.collected_count}/{NUM_IMAGES}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.imshow("Calibration", display_img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') and ret:
                print("\n" + "-"*30)
                self.save_frame_manual(corners2)
            elif key == ord('c'):
                if self.collected_count < 3:
                    print(f"\n[WARN] 数据不足！")
                else:
                    self.solve_handeye()
                    self.running = False
                    break
            elif key == ord('q'):
                self.running = False
                break

            if self.collected_count >= NUM_IMAGES:
                self.solve_handeye()
                self.running = False
                break

        self.cleanup()

    def save_frame_manual(self, corners):
        # 1. 计算 Target -> Camera
        ret, rvec, tvec = cv2.solvePnP(self.objp, corners, self.K, self.dist_coeffs)
        if not ret: return

        # 2. 用户输入 (Base -> Gripper)
        while True:
            try:
                print("请输入 tf2_echo 的原始数据 [x y z roll pitch yaw]:")
                user_input = input(">> ")
                if user_input.lower() == 'c': return
                vals = list(map(float, user_input.strip().split()))
                if len(vals) != 6: continue
                x, y, z, r, p, yw = vals
                break
            except ValueError:
                pass

        # 3. 🔴 关键修正：将 Base->Gripper 取反为 Gripper->Base
        
        # 3.1 构造 4x4 变换矩阵 (T_base_to_gripper)
        R_b2g = self.euler_to_matrix(r, p, yw)
        T_b2g = np.eye(4)
        T_b2g[:3, :3] = R_b2g
        T_b2g[:3, 3] = np.array([x, y, z])

        # 3.2 矩阵求逆 (得到 T_gripper_to_base)
        T_g2b = np.linalg.inv(T_b2g)

        # 3.3 提取旋转和平移
        R_g2b = T_g2b[:3, :3]
        t_g2b = T_g2b[:3, 3].reshape(3, 1)

        # 4. 保存取反后的数据
        R_target2cam_mat, _ = cv2.Rodrigues(rvec)
        
        self.R_gripper2base.append(R_g2b)  # 存入取反后的 R
        self.t_gripper2base.append(t_g2b)  # 存入取反后的 t
        self.R_target2cam.append(R_target2cam_mat)
        self.t_target2cam.append(tvec)

        self.collected_count += 1
        print(f"✅ 第 {self.collected_count} 组保存成功 (已自动执行矩阵求逆)")

    def solve_handeye(self):
        print("\n正在计算标定结果...")
        try:
            # 输入已经是 Gripper2Base，符合 OpenCV 这种调用方式
            R_cam2base, t_cam2base = cv2.calibrateHandEye(
                self.R_gripper2base, self.t_gripper2base,
                self.R_target2cam, self.t_target2cam,
                method=cv2.CALIB_HAND_EYE_TSAI
            )
            print("="*40)
            print("       🏆 最终结果 (Camera in Base)       ")
            print("="*40)
            print("平移向量 T (meters):\n", t_cam2base)
            print("  -> X: %.4f" % t_cam2base[0])
            print("  -> Y: %.4f" % t_cam2base[1])
            print("  -> Z: %.4f" % t_cam2base[2])
            
            dist = np.linalg.norm(t_cam2base)
            print(f"\n📏 相机距离基座的直线距离: {dist:.4f} 米")
            print("="*40)
            np.savez(OUTPUT_FILE.replace('.npy', '.npz'), R=R_cam2base, T=t_cam2base)
        except Exception as e:
            print(f"[ERROR] {e}")

    def cleanup(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()
        self.destroy_node()
        rclpy.shutdown()

def main():
    rclpy.init()
    node = EyeToHandCalibration()
    node.thread.join()

if __name__ == "__main__":
    main()
