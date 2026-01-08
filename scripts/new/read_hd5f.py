import h5py

def print_h5_structure(name, obj):
    """递归打印 HDF5 文件的层级结构"""
    if isinstance(obj, h5py.Dataset):
        print(f"📊 Dataset: {name}, shape={obj.shape}, dtype={obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"📁 Group: {name}")

with h5py.File("/share/project/caomingyu/a2d_data/675/A2D0015AC00608/22510/aligned_joints.h5", "r") as f:
    f.visititems(print_h5_structure)
    # print(f["/observations/joint_states/arm_left_1/joint_names"][()])
    # print(f["/observations/joint_states/arm_right/joint_names"][()])
