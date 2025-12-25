import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from bt_task_manager_ros2.moveit_arm import ArmMovetoPoseInCartesian, ArmMoveToName
import time
import tf2_ros
from tf2_geometry_msgs import do_transform_pose
import math
from py_trees.common import Status

gen3_config = {
    "group_name": "right_arm",
    "base_link_name": "base_link",
    "joint_names": ["right_joint_1", "right_joint_2", "right_joint_3", 
                    "right_joint_4", "right_joint_5", "right_joint_6", "right_joint_7"],
    "end_effector_name": "right_end_effector_link",
    "planner_id": "RRTConnectkConfigDefault",
    "max_velocity": 0.2,
    "max_acceleration": 0.2,
    "named_poses": {
        "zero": [0, 0, 0.5, 0, 0, 0, 0],
        "home": [0, 0, 0.5, 0, 0, 0, 0],
        "reset_pose": [0, 0, 0.5, 0, 0, 0, 0]
    },
}

class ArmMovementNode(Node):
    def __init__(self):
        super().__init__("arm_movement_node")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # 分别订阅A和B点的数据
        self.sub_a = self.create_subscription(
            PoseStamped,
            "/global/position_a",
            self.a_callback,
            10
        )
        self.sub_b = self.create_subscription(
            PoseStamped,
            "/global/position_b",
            self.b_callback,
            10
        )
        
        self.a_data = None  # 缓存A点数据
        self.b_data = None  # 缓存B点数据
        self.is_executing = False
        self.get_logger().info("机械臂运动节点启动（分接收模式）...")

    def a_callback(self, msg):
        """接收A点数据并缓存"""
        self.a_data = msg
        self.get_logger().info(
            f"收到A点：位置({msg.pose.position.x:.3f},{msg.pose.position.y:.3f},{msg.pose.position.z:.3f})"
        )
        self.check_and_execute()  # 检查是否可以执行

    def b_callback(self, msg):
        """接收B点数据并缓存"""
        self.b_data = msg
        self.get_logger().info(
            f"收到B点：位置({msg.pose.position.x:.3f},{msg.pose.position.y:.3f},{msg.pose.position.z:.3f})"
        )
        self.check_and_execute()  # 检查是否可以执行

    def check_and_execute(self):
        """当A、B点数据都收到且未在执行时，触发轨迹执行"""
        if self.a_data and self.b_data and not self.is_executing:
            self.get_logger().info("A、B点数据齐全，开始执行轨迹...")
            self.execute_trajectory()

    def transform_base_to_tool(self, base_pose):
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = gen3_config["base_link_name"]
        # 关键修复：将 Time 转换为 ROS 2 标准的 Time 消息类型
        pose_stamped.header.stamp = rclpy.time.Time(seconds=0).to_msg()  # 使用 to_msg() 转换
        pose_stamped.pose.position.x = base_pose[0]
        pose_stamped.pose.position.y = base_pose[1]
        pose_stamped.pose.position.z = base_pose[2]
        pose_stamped.pose.orientation.w = 1.0
        try:
            transformed_pose = self.tf_buffer.transform(
                pose_stamped,
                gen3_config["end_effector_name"],
                timeout=rclpy.duration.Duration(seconds=2.0)
            )
            return [
                transformed_pose.pose.position.x,
                transformed_pose.pose.position.y,
                transformed_pose.pose.position.z
            ]
        except tf2_ros.TransformException as e:
            self.get_logger().error(f"TF坐标变换失败：{e}")
            return None

    def calculate_rotation_adjustment(self, a_yaw, b_yaw):
        # 保持原逻辑不变
        a_yaw = a_yaw % 360
        b_yaw = b_yaw % 360
        delta = (b_yaw - a_yaw) % 360
        return delta - 360 if delta > 180 else delta

    def execute_trajectory(self):
        """
        替换用：把 A/B 点拼出 steps，然后把 steps 变成行为实例放进 Sequence.children，
        并使用 py_trees_ros BehaviourTree 的 setup + tick_tock 去执行（完全借鉴 ex_moveit_arm.py）。
        - 不新增任何外部 config（只用已有的 gen3_config）
        - children 中的 goal 使用变量（A_pos/A_euler/A_above/...），每次数据自动变化
        """
        global task_end
        self.is_executing = True
        task_end = False

        try:
            # 解析 A/B（直接使用基座坐标系下的数据；保持你原有的 orientation.x/y/z -> euler 逻辑）
            a_msg = self.a_data
            b_msg = self.b_data

            A_pos = [a_msg.pose.position.x, a_msg.pose.position.y, a_msg.pose.position.z]
            A_roll = a_msg.pose.orientation.x
            A_pitch = a_msg.pose.orientation.y
            A_yaw = a_msg.pose.orientation.z
            A_euler = [A_roll, A_pitch, A_yaw]

            B_pos = [b_msg.pose.position.x, b_msg.pose.position.y, b_msg.pose.position.z]
            B_roll = b_msg.pose.orientation.x
            B_pitch = b_msg.pose.orientation.y
            B_yaw = b_msg.pose.orientation.z
            B_euler = [B_roll, B_pitch, B_yaw]

            self.get_logger().info(
                "A(base): {} euler:{} | B(base): {} euler:{}".format(
                    [round(v,4) for v in A_pos],
                    [round(v,3) for v in A_euler],
                    [round(v,4) for v in B_pos],
                    [round(v,3) for v in B_euler],
                )
            )

            # 计算朝向调整并构造上方点（变量化）
            delta_yaw = self.calculate_rotation_adjustment(A_euler[2], B_euler[2])
            adjusted_B_euler = [B_roll, B_pitch, A_euler[2] + delta_yaw] 

            A_above = A_pos.copy()
            A_above[2] += 0.05
            B_above = B_pos.copy()
            B_above[2] += 0.05

            # 构造 steps 列表（使用变量作为 goal）
            steps = [
                ("A_Above", ArmMovetoPoseInCartesian, A_above + A_euler),
                ("A_Point", ArmMovetoPoseInCartesian, A_pos + A_euler),
                ("A_Up", ArmMovetoPoseInCartesian, A_above + A_euler),

                ("B_Above", ArmMovetoPoseInCartesian, B_above + A_euler),
                ("B_Rotate", ArmMovetoPoseInCartesian, B_above + adjusted_B_euler),
                ("B_Point", ArmMovetoPoseInCartesian, B_pos + adjusted_B_euler),
                ("B_Up", ArmMovetoPoseInCartesian, B_above + adjusted_B_euler),

                ("Back_Home", ArmMoveToName, "home"),
            ]

            # 把 steps 转成行为实例（children），注意这里使用的是位置参数构造：behavior_cls(name, gen3_config, goal)
            children = []
            for (name_str, behavior_cls, goal) in steps:
                beh = behavior_cls(name_str, gen3_config, goal)
                children.append(beh)

            # 创建主序列（Sequence）并构建行为树 —— 与 ex_moveit_arm.py 中的 main_sequence / BehaviourTree 对应
            main_sequence = py_trees.composites.Sequence(
                name="main_sequence",
                memory=True,
                children=children
            )
            tree = py_trees_ros.trees.BehaviourTree(root=main_sequence, unicode_tree_debug=False)

            # 与 ex_moveit_arm.py 完全相同的 setup + tick_tock 流程（blocking，直到树 shutdown）
            try:
                tree.setup(node=self, timeout=15)
                tree.tick_tock(period_ms=100.0, post_tick_handler=post_tick_handler)
            except py_trees_ros.exceptions.TimedOutError as e:
                self.get_logger().error("failed to setup the tree, aborting [{}]".format(str(e)))
                tree.shutdown()
                return
            except KeyboardInterrupt:
                self.get_logger().error("tree setup interrupted")
                tree.shutdown()
                return

            # 等待短暂确保 tree 完全结束
            time.sleep(0.05)
            self.get_logger().info("行为树执行返回")

        except Exception as e:
            self.get_logger().error("execute_trajectory 出错: {}".format(e))
            # 失败时尝试用行为树做回退（同样传入变量/字符串构建 steps_reset）
            try:
                steps_reset = [("Error_Reset", ArmMoveToName, "reset_pose")]
                children_reset = [ ArmMoveToName("Error_Reset", gen3_config, "reset_pose") ]
                root_reset = py_trees.composites.Sequence(name="reset_seq", memory=True, children=children_reset)
                tree_reset = py_trees_ros.trees.BehaviourTree(root=root_reset, unicode_tree_debug=False)
                try:
                    tree_reset.setup(node=self, timeout=5)
                    tree_reset.tick_tock(period_ms=100.0, post_tick_handler=post_tick_handler)
                except Exception as ee:
                    self.get_logger().info("回退行为树执行失败: {}".format(ee))
                finally:
                    try:
                        tree_reset.shutdown()
                    except Exception:
                        pass
            except Exception as ee:
                self.get_logger().info("回退异常: {}".format(ee))

        finally:
            # 清理状态（保持原名）
            self.is_executing = False
            self.a_data = None
            self.b_data = None
            self.get_logger().info("等待下一组 A/B 点...")

def main(args=None):
    rclpy.init(args=args)
    node = ArmMovementNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
