import numpy as np

# 加载你的标定文件
file_path = '/home/rpp/robot_ws/handeye_result_eye_to_hand.npz'

try:
    data = np.load(file_path)
    R = data['R']
    T = data['T']

    print("\n" + "="*30)
    print("📂 手眼标定矩阵内容查看")
    print("="*30)
    
    print("\n[旋转矩阵 R] (3x3):")
    print(R)
    
    print("\n[平移向量 T] (单位: 米):")
    print(f"X: {T[0][0]:.4f} m  ({T[0][0]*100:.2f} cm)")
    print(f"Y: {T[1][0]:.4f} m  ({T[1][0]*100:.2f} cm)")
    print(f"Z: {T[2][0]:.4f} m  ({T[2][0]*100:.2f} cm)")
    
    print("\n[验证建议]:")
    print(f"这个 T 向量代表相机中心距离机器人基座原点的距离。")
    print(f"请拿卷尺大概量一下，相机到机器人的直线距离是否接近 {np.linalg.norm(T):.2f} 米。")
    print("="*30 + "\n")

except Exception as e:
    print(f"❌ 读取失败: {e}")