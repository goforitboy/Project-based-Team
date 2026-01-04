#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import sys
import os
import time

from servo_controller.pca9685_i2c_bridge import (
    I2CBridge,
    set_pca9685_pwm_freq,
    set_pca9685_channel_pwm
)

from kortex_api.autogen.messages import InterconnectConfig_pb2


# -----------------------------
# 角度 → PCA9685 PWM ticks
# -----------------------------
def angle_to_pwm_ticks(angle_deg: float) -> int:
    angle = max(0.0, min(180.0, angle_deg))
    pulse_min = 0.5   # ms
    pulse_max = 2.5   # ms
    period = 20.0     # ms
    pulse_ms = pulse_min + (angle / 180.0) * (pulse_max - pulse_min)
    ticks = int(pulse_ms / period * 4096)
    return max(0, min(4095, ticks))


# -----------------------------
# ROS2 节点
# -----------------------------
class ServoControllerNode(Node):
    def __init__(self):
        super().__init__('servo_controller')
        self.get_logger().info("Starting servo_controller node...")

        # ===== Kinova utilities =====
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        import utilities

        args = utilities.parseConnectionArguments()
        self.device_connection = utilities.DeviceConnection.createTcpConnection(args)
        self.router = self.device_connection.__enter__()
        self.get_logger().info("Connected to Kinova.")

        # ===== I2C Bridge =====
        self.bridge = I2CBridge(self.router)
        self.pca_addr = 0x40

        # ===== 配置 Kinova Interconnect I2C =====
        self.bridge.Configure(
            True,
            InterconnectConfig_pb2.I2C_MODE_FAST,
            InterconnectConfig_pb2.I2C_DEVICE_ADDRESSING_7_BITS
        )
        time.sleep(0.2)
        self.get_logger().info("Interconnect I2C configured.")

        # ===== PCA9685 初始化 =====
        set_pca9685_pwm_freq(self.bridge, self.pca_addr, 50)
        time.sleep(0.05)
        self.get_logger().info("PCA9685 PWM frequency set to 50Hz.")

        # MODE2 推挽输出
        self.bridge.WriteValue(self.pca_addr, 0x01, 0x04)

        # 初始化舵机到 90°
        init_angle = 90.0
        ticks = angle_to_pwm_ticks(init_angle)
        set_pca9685_channel_pwm(self.bridge, self.pca_addr, 1, ticks)  # SG90
        set_pca9685_channel_pwm(self.bridge, self.pca_addr, 2, ticks)  # MG995
        self.get_logger().info(f"Servos initialized to {init_angle} degrees.")

        # ===== ROS2 订阅 =====
        self.create_subscription(
            Float32,
            '/servo/sg90/angle',
            self.sg90_callback,
            10
        )
        self.create_subscription(
            Float32,
            '/servo/mg995/angle',
            self.mg995_callback,
            10
        )

    # -----------------------------
    # 回调函数
    # -----------------------------
    def sg90_callback(self, msg: Float32):
        ticks = angle_to_pwm_ticks(msg.data)
        set_pca9685_channel_pwm(self.bridge, self.pca_addr, 1, ticks)
        self.get_logger().info(f"SG90 angle set to {msg.data:.1f}°")

    def mg995_callback(self, msg: Float32):
        ticks = angle_to_pwm_ticks(msg.data)
        set_pca9685_channel_pwm(self.bridge, self.pca_addr, 2, ticks)
        self.get_logger().info(f"MG995 angle set to {msg.data:.1f}°")

    # -----------------------------
    # 节点销毁
    # -----------------------------
    def destroy_node(self):
        self.get_logger().info("Closing Kinova connection...")
        try:
            self.device_connection.__exit__(None, None, None)
        except Exception as e:
            self.get_logger().warn(f"Error closing connection: {e}")
        super().destroy_node()


# -----------------------------
# main
# -----------------------------
def main(args=None):
    rclpy.init(args=args)
    node = ServoControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
