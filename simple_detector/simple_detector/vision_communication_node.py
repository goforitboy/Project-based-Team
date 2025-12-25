# vision_communication_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import copy

class VisionCommunicationNode(Node):
    def __init__(self):
        super().__init__("vision_communication_node")
        # 初始化A、B点数据缓存n
        self.a_data = None  # 存储A点PoseStamped
        self.b_data = None  # 存储B点PoseStamped
        
        # 分别订阅A、B点的原始数据（两个发送者）
        self.sub_a = self.create_subscription(
            PoseStamped,
            "/vision/position_a",  # A点发送者话题
            self.a_callback,
            10
        )
        self.sub_b = self.create_subscription(
            PoseStamped,
            "/vision/position_b",  # B点发送者话题
            self.b_callback,
            10
        )
        
        # 分开发布A和B点数据
        self.pub_a = self.create_publishner(PoseStamped, "/global/position_a", 10)
        self.pub_b = self.create_publisher(PoseStamped, "/global/position_b", 10)
        self.feedback_pub = self.create_publisher(String, "/vision/feedback", 10)
        self.get_logger().info("视觉通信节点启动（分开发送模式）...")

    def a_callback(self, msg):
        """接收A点数据并直接发布"""
        self.a_data = msg
        self.pub_a.publish(msg)
        self.get_logger().info(
            f"收到并发布A点：位置({msg.pose.position.x:.3f},{msg.pose.position.y:.3f},{msg.pose.position.z:.3f})"
        )
        self.feedback_pub.publish(String(data=f"A点已发布: {msg.pose.position}"))

    def b_callback(self, msg):
        """接收B点数据并直接发布"""
        self.b_data = msg
        self.pub_b.publish(msg)
        self.get_logger().info(
            f"收到并发布B点：位置({msg.pose.position.x:.3f},{msg.pose.position.y:.3f},{msg.pose.position.z:.3f})"
        )
        self.feedback_pub.publish(String(data=f"B点已发布: {msg.pose.position}"))

def main(args=None):
    rclpy.init(args=args)
    node = VisionCommunicationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()