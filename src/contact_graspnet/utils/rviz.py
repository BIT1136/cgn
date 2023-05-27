import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Pose

import rospy
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Pose, Vector3
from visualization_msgs.msg import MarkerArray, Marker


draw_grasp_id = 0
draw_cam_id = 0


def draw_grasp(pub, pose, frame):
    global draw_grasp_id
    markers = create_grasp_markers(
        frame, pose, [0.5, 1, 0.5, 1], "grasp", draw_grasp_id
    )
    draw_grasp_id += 4
    pub.publish(MarkerArray(markers=markers))


def clear_markers(pub):
    delete = [Marker(action=Marker.DELETEALL)]
    pub.publish(MarkerArray(delete))


def create_grasp_markers(frame, pose: Pose, color, ns, id=0):
    # 抓取点位于指尖
    pose_mat = pose_msg_to_matrix(pose)
    w, d, radius = 0.075, 0.05, 0.005

    left_point = np.dot(pose_mat, np.array([-w / 2, 0, -d / 2, 1]))
    left_pose = pose_mat.copy()
    left_pose[:3, 3] = left_point[:3]
    scale = [radius, radius, d]
    left = create_marker(Marker.CYLINDER, frame, left_pose, scale, color, ns, id)

    right_point = np.dot(pose_mat, np.array([w / 2, 0, -d / 2, 1]))
    right_pose = pose_mat.copy()
    right_pose[:3, 3] = right_point[:3]
    scale = [radius, radius, d]
    right = create_marker(Marker.CYLINDER, frame, right_pose, scale, color, ns, id + 1)

    wrist_point = np.dot(pose_mat, np.array([0.0, 0.0, -d * 5 / 4, 1]))
    wrist_pose = pose_mat.copy()
    wrist_pose[:3, 3] = wrist_point[:3]
    scale = [radius, radius, d / 2]
    wrist = create_marker(Marker.CYLINDER, frame, wrist_pose, scale, color, ns, id + 2)

    palm_point = np.dot(pose_mat, np.array([0.0, 0.0, -d, 1]))
    palm_pose = pose_mat.copy()
    palm_pose[:3, 3] = palm_point[:3]
    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_rotvec([0, np.pi / 2, 0]).as_matrix()
    palm_pose = np.dot(palm_pose, rot)
    scale = [radius, radius, w]
    palm = create_marker(Marker.CYLINDER, frame, palm_pose, scale, color, ns, id + 3)

    return [left, right, wrist, palm]


def create_marker(type, frame, pose, scale=[1, 1, 1], color=(1, 1, 1, 1), ns="", id=0):
    if np.isscalar(scale):
        scale = [scale, scale, scale]
    msg = Marker()
    msg.header.frame_id = frame
    msg.header.stamp = rospy.Time()
    msg.ns = ns
    msg.id = id
    msg.type = type
    msg.action = Marker.ADD
    msg.pose = matrix_to_pose_msg(pose)
    msg.scale = Vector3(*scale)
    msg.color = ColorRGBA(*color)
    return msg


def pose_msg_to_matrix(pose_msg: Pose):
    """将 ROS 的 Pose 消息转换为变换矩阵"""
    translation = np.array(
        [pose_msg.position.x, pose_msg.position.y, pose_msg.position.z]
    )
    rotation = np.array(
        [
            pose_msg.orientation.x,
            pose_msg.orientation.y,
            pose_msg.orientation.z,
            pose_msg.orientation.w,
        ]
    )
    # from_quat(x, y, z, w)
    rotation_matrix = Rotation.from_quat(rotation).as_matrix()
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix[:3, :3]
    transform_matrix[:3, 3] = translation
    return transform_matrix


def matrix_to_pose_msg(pose_mat):
    """将变换矩阵转换为 ROS 的 Poe 消息"""
    pose = Pose()
    # as_quat()->(x, y, z, w)
    translation = Rotation.from_matrix(pose_mat[:3, :3]).as_quat()
    pose.orientation.x = translation[0]
    pose.orientation.y = translation[1]
    pose.orientation.z = translation[2]
    pose.orientation.w = translation[3]
    pose.position.x = pose_mat[0, 3]
    pose.position.y = pose_mat[1, 3]
    pose.position.z = pose_mat[2, 3]
    return pose
