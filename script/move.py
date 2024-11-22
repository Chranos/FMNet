import os
import shutil

def move_matching_files(source_folder, match_folder, destination_folder):
    """
    将 source_folder 中所有与 match_folder 同名（忽略后缀名）的文件移动到 destination_folder。
    如果目标文件夹不存在，会自动创建。

    :param source_folder: 源文件夹路径（a 文件夹）
    :param match_folder: 匹配文件夹路径（b 文件夹）
    :param destination_folder: 目标文件夹路径（c 文件夹）
    """
    # 检查源和匹配文件夹是否存在
    if not os.path.exists(source_folder):
        print(f"源文件夹 '{source_folder}' 不存在。")
        return
    if not os.path.exists(match_folder):
        print(f"匹配文件夹 '{match_folder}' 不存在。")
        return

    # 如果目标文件夹不存在，则创建
    os.makedirs(destination_folder, exist_ok=True)

    # 获取匹配文件夹中所有文件的名字（忽略后缀）
    match_files = set(
        os.path.splitext(f)[0] for f in os.listdir(match_folder)
        if os.path.isfile(os.path.join(match_folder, f))
    )

    # 遍历源文件夹，检查是否匹配
    files_moved = 0
    for file_name in os.listdir(source_folder):
        source_path = os.path.join(source_folder, file_name)

        if os.path.isfile(source_path):
            name_without_extension = os.path.splitext(file_name)[0]  # 去除后缀名

            if name_without_extension in match_files:
                destination_path = os.path.join(destination_folder, file_name)
                shutil.move(source_path, destination_path)
                files_moved += 1
                print(f"已移动: {file_name} -> {destination_path}")

    print(f"共移动了 {files_moved} 个匹配的文件到 '{destination_folder}'。")

# 示例调用
source_folder = "/workspace/CODlab/data/datasets/COD10K_train/test/GT"  # 替换为 a 文件夹的实际路径
match_folder = "/workspace/CODlab/data/datasets/COD10K_train/train/Edge"   # 替换为 b 文件夹的实际路径
destination_folder = "/workspace/CODlab/data/datasets/COD10K_train/train/GT"  # 替换为 c 文件夹的实际路径

move_matching_files(source_folder, match_folder, destination_folder)