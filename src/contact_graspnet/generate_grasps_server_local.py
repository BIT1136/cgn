#!/usr/bin/env python3

import os
import sys
import argparse
import time
import numpy as np
import rospy
import tensorflow.compat.v1 as tf
import config_utils
import rospkg
import ros_numpy
from scipy.spatial.transform import Rotation

from cgn.srv import GenerateGrasps, GenerateGraspsResponse
from geometry_msgs.msg import PoseArray, Pose, PointStamped
from std_msgs.msg import Header
from utils.transformations import *  # python3 can't import tf.transformations
from copy import deepcopy
from contact_grasp_estimator import GraspEstimator
from contact_graspnet.utils.rviz import draw_grasp, clear_markers
from visualization_msgs.msg import MarkerArray

# ---------------------
# Set tensorflow params
# ---------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))

tf.disable_eager_execution()
tf.disable_v2_behavior()
physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Create a session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
# config.gpu_options.per_process_gpu_memory_fraction=0.2 # applies to my GPU -- 3060 RTX
sess = tf.Session(config=config)

# ---------------------------------
# set contact graspnet model config
# ---------------------------------
model_path = os.path.join(rospkg.RosPack().get_path("cgn"), "models")
# print(model_path)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ckpt_dir",
    default=model_path + "/scene_test_2048_bs3_hor_sigma_001",
    help="Log dir [default: ../../models/scene_test_2048_bs3_hor_sigma_001]",
)
parser.add_argument(
    "--local_regions",
    action="store_true",
    default=False,
    help="Crop 3D local regions around given segments.",
)
parser.add_argument(
    "--filter_grasps",
    action="store_true",
    default=False,
    help="Filter grasp contacts according to segmap.",
)
parser.add_argument(
    "--forward_passes",
    type=int,
    default=5,
    help="Run multiple parallel forward passes to mesh_utils more potential contact points.",
)
parser.add_argument(
    "--arg_configs", nargs="*", type=str, default=[], help="overwrite config parameters"
)
parser.add_argument(
    "__name", nargs="*", type=str, default=[], help="overwrite config parameters"
)  # this parameter simply allows ROS to launch the file with roslaunch
FLAGS = parser.parse_args()

global_config = config_utils.load_config(
    FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs
)

# print(str(global_config))
# print('pid: %s'%(str(os.getpid())))

checkpoint_dir = FLAGS.ckpt_dir
print("载入检查点：", checkpoint_dir)

# Build the model
grasp_estimator = GraspEstimator(global_config)
grasp_estimator.build_network()

# Add ops to save and restore all the variables.
saver = tf.train.Saver(save_relative_paths=True)

marker_pub = rospy.Publisher("marker", MarkerArray, queue_size=1)

print("加载权重")
grasp_estimator.load_weights(sess, saver, checkpoint_dir, mode="test")

# 第一次预测会很慢
a,b,c,d=grasp_estimator.predict_scene_grasps(
        sess,
        pc_full=np.random.rand(1000, 3),
        pc_segments={0:np.random.rand(100, 3)},
        local_regions=True,
        filter_grasps=True,
        forward_passes=FLAGS.forward_passes,
    )


# callback function in generating grasps service
def generate_grasps(req):
    """
    Arguments:
        req {GenerateGrasps} -- message container full_pcl and objs_pcl
    Return:
        response {GenerateGraspsResponse} -- number_objects, all_grasp_poses, grasp_contact_points, object_heights, all_scores
    """

    frame_id = req.full_pcl.header.frame_id

    # reshape pcl to be in the right form to be processed
    pcl = ros_numpy.numpify(req.full_pcl)
    pcl_np = np.concatenate(
        (pcl["x"].reshape(-1, 1), pcl["y"].reshape(-1, 1), pcl["z"].reshape(-1, 1)),
        axis=1,
    )

    # reshape each object pcl
    objs_np = {}
    for i, object_pcl in enumerate(req.objects_pcl):
        obj_np = ros_numpy.numpify(object_pcl)
        obj_np = np.concatenate(
            (
                obj_np["x"].reshape(-1, 1),
                obj_np["y"].reshape(-1, 1),
                obj_np["z"].reshape(-1, 1),
            ),
            axis=1,
        )
        objs_np[i] = obj_np

    response = GenerateGraspsResponse()

    print("预测抓取...")
    t_start = time.perf_counter()
    pred_grasps_cam, scores, contacts, openings = grasp_estimator.predict_scene_grasps(
        sess,
        pc_full=pcl_np,
        pc_segments=objs_np,
        local_regions=True,
        filter_grasps=True,
        forward_passes=FLAGS.forward_passes,
    )

    # loop through each object's pcl and adjust predicted grasps
    for i in range(len(req.objects_pcl)):

        if i in pred_grasps_cam.keys() and len(pred_grasps_cam[i])>0:
            pose_array = pred_grasps_cam[i]

            # 取前5个抓取位姿
            this_scores = scores[i]
            idx = np.argsort(this_scores)[::-1][-5:]
            pose_array = pose_array[idx]
            this_scores = this_scores[idx]
            print(f"最高分：{this_scores[0]:.3f}")

            pose_array = forward_poses(pose_array, 0.1)
            pose_array = to_pose_array_msg(pose_array, frame_id)

            # 绘制抓取位姿
            clear_markers(marker_pub)
            for pose,q in zip(pose_array.poses,this_scores):
                draw_grasp(marker_pub, pose, frame_id,q)
                time.sleep(0.1)

            response.all_grasp_poses.append(pose_array)
            
            response.all_scores = response.all_scores + this_scores.tolist()
            if np.isnan(np.nanmean(scores[i])):
                print(scores[i])
                print(np.nanmean(scores[i]))
                print("nan scores: {}, object: {}".format(scores[i], i))

        else:  # empty grasps
            response.all_grasp_poses.append(PoseArray())
            response.all_scores.append(0.0)

    t_end = time.perf_counter()
    print(f"抓取已生成,耗时: {(t_end - t_start)*1000:.2f}ms")
    return response


def to_point_stamped(points, frame_id):
    s = []
    for p in points:
        point_stamped = PointStamped()
        point_stamped.header.frame_id = frame_id
        point_stamped.header.stamp = rospy.get_rostime()
        point_stamped.point.x = p[0]
        point_stamped.point.y = p[1]
        point_stamped.point.z = p[2]
        s.append(point_stamped)
    return s


def forward_poses(poses, length):
    for i in range(len(poses)):
        point = np.dot(poses[i], np.array([0.0, 0.0, length, 1]))
        poses[i, :3, 3] = point[:3]
    return poses


def to_pose_array_msg(poses, frame):
    pose_array = PoseArray(header=Header(frame_id=frame, stamp=rospy.get_rostime()))
    for pose_mat in poses:
        pose = Pose()
        translation = Rotation.from_matrix(pose_mat[:3, :3]).as_quat()
        pose.orientation.x = translation[0]
        pose.orientation.y = translation[1]
        pose.orientation.z = translation[2]
        pose.orientation.w = translation[3]
        pose.position.x = pose_mat[0, 3]
        pose.position.y = pose_mat[1, 3]
        pose.position.z = pose_mat[2, 3]
        pose_array.poses.append(pose)
    return pose_array


if __name__ == "__main__":
    rospy.init_node("cgn_server", anonymous=True)
    try:
        s = rospy.Service(
            f"{rospy.get_name()}/generate_grasps", GenerateGrasps, generate_grasps
        )
        print("cgn_server就绪")
    except KeyboardInterrupt:
        sys.exit()
    rospy.spin()
