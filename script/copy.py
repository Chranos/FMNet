import os
import shutil

def copy_files(source_folder, destination_folder):
    """
    将 source_folder 文件夹中的所有文件复制到 destination_folder 文件夹中。
    如果目标文件夹不存在，会自动创建。

    :param source_folder: 源文件夹路径
    :param destination_folder: 目标文件夹路径
    """
    # 检查源文件夹是否存在
    if not os.path.exists(source_folder):
        print(f"源文件夹 {source_folder} 不存在。")
        return

    # 如果目标文件夹不存在，则创建
    os.makedirs(destination_folder, exist_ok=True)

    # 获取源文件夹中的所有文件
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    if not files:
        print(f"源文件夹 {source_folder} 中没有文件。")
        return

    # 遍历并复制文件
    for file_name in files:
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)

        # 复制文件
        shutil.copy2(source_path, destination_path)  # 使用 copy2 保留元数据
        print(f"已复制: {file_name} -> {destination_path}")

    print("所有文件已成功复制。")

# 示例调用
source_folder = "/workspace/CODlab/data/datasets/COD10K_train/test/Imgs"  # 替换为你的源文件夹路径
destination_folder = "/workspace/CODlab/data/datasets/COD10K_test/Imgs"  # 替换为你的目标文件夹路径

copy_files(source_folder, destination_folder)