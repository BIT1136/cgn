# CGN

代码修改自[这里](https://github.com/jucamohedano/ros_contact_graspnet)，原论文为[Contact-GraspNet](https://github.com/NVlabs/contact_graspnet)。使用black格式化代码。

## 安装环境

    conda create -c conda-forge -n cgn python=3.7 numpy=1.19 opencv=4.4 tensorflow=2.2 tensorboard=2.2 trimesh=3.8 h5py=2.10 mayavi=4.7 matplotlib=3.3 tqdm=4.51 pyyaml=5.3 pyrender importlib_resources

重新编译pointnet2 tf_ops：

    sh compile_pointnet_tfops.sh

## 下载模型检查点

从[这里](https://drive.google.com/drive/folders/1tBHKf60K8DLM5arm-Chyf7jxkzOr5zGl)下载一个文件夹（默认使用scene_test_2048_bs3_hor_sigma_0025，在generate_grasps_server_local.py第55行修改）到models文件夹中，将`config.yaml`中的`model: contact_graspnet`修改为`model: contact_graspnet_model`。修改其中checkpoint文件的第一行以加载不同的检查点。

## 运行节点

    roslaunch cgn node.launch

提供服务/cgn_server/generate_grasps，将预测出的抓取MarkerArray发布到cgn_marker。
