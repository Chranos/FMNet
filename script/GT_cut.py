import os
import shutil

def get_file_stem(file_name):
    """获取文件名的主干部分（忽略后缀）"""
    return os.path.splitext(file_name)[0]


# 文件夹路径
folder_a = '/workspace/CODlab/data/datasets/CAMO/train/GT'  # A 文件夹路径
folder_b = '/workspace/CODlab/data/datasets/CAMO_1/Images/Test'  # B 文件夹路径
folder_c = '/workspace/CODlab/data/datasets/CAMO/train/Edgeall'  # C 文件夹路径
folder_c_a = '/workspace/CODlab/data/datasets/CAMO/train/Edge'  # C_A 文件夹路径
folder_c_b = '/workspace/CODlab/data/datasets/CAMO/test/GT'  # C_B 文件夹路径

# 创建 C_A 和 C_B 文件夹
# os.makedirs(folder_c_a, exist_ok=True)
# os.makedirs(folder_c_b, exist_ok=True)

# 获取 A 和 B 文件夹中的文件名主干集合
stems_a = {get_file_stem(file_name) for file_name in os.listdir(folder_a) if file_name.lower().endswith('.png')}
stems_b = {get_file_stem(file_name) for file_name in os.listdir(folder_b) if file_name.lower().endswith('.jpg')}

# 遍历 C 文件夹中的文件
for file_name in os.listdir(folder_c):
    if file_name.lower().endswith('.jpg'):  # 只处理 png 文件
        file_stem = get_file_stem(file_name)
        source_path = os.path.join(folder_c, file_name)

        # 检查是否与 A 或 B 文件夹中的文件主干匹配
        if file_stem in stems_a:
            shutil.move(source_path, os.path.join(folder_c_a, file_name))
        # elif file_stem in stems_b:
        #     shutil.move(source_path, os.path.join(folder_c_b, file_name))

print("文件分类完成！")