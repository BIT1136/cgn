# CGN

代码修改自[6 DoF grasp synthesis project](https://github.com/jucamohedano/ros_contact_graspnet)，原模型仓库为[Contact-GraspNet](https://github.com/NVlabs/contact_graspnet)。

## 安装环境

    conda create -c conda-forge -n cgn python=3.7 scipy pyyaml importlib_resources rospkg
    pip install tensorflow==2.5

重新编译pointnet2算子：

    sh compile_pointnet_tfops.sh

## 下载模型检查点

从[这里](https://drive.google.com/drive/folders/1tBHKf60K8DLM5arm-Chyf7jxkzOr5zGl)下载一个文件夹（默认使用scene_test_2048_bs3_hor_sigma_0025，在generate_grasps_server_local.py第55行修改）到models文件夹中，将`config.yaml`中的`model: contact_graspnet`修改为`model: contact_graspnet_model`。修改其中checkpoint文件的第一行以加载不同的检查点。

## 运行节点

    roslaunch cgn node.launch

提供服务/cgn_server/generate_grasps，将预测出的抓取MarkerArray发布到cgn_marker。

## 参考

[Deep Grasping ROS](https://github.com/gist-ailab/deep-grasping)也包装了Contact-GraspNet。
