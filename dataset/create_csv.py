import os
import csv


def save_image_names_to_csv(folder_path, output_csv):
    """
    整合指定文件夹下的图片名称及后缀到 CSV 文件中。

    :param folder_path: 包含图片的文件夹路径
    :param output_csv: 输出的 CSV 文件路径
    """
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"错误：文件夹 {folder_path} 不存在！")
        return

    # 创建或覆盖 CSV 文件
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        # 写入标题行
        writer.writerow(['File Name', 'Extension'])

        # 遍历文件夹中的文件
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            # 检查是否为文件且后缀匹配
            if os.path.isfile(file_path):
                name, ext = os.path.splitext(file_name)
                if ext.lower() in image_extensions:
                    # 写入文件名和后缀
                    writer.writerow([name, ext])

    print(f"图片信息已保存到 {output_csv}")


# 使用示例
folder_path = r"../../datasets/MSRS/train/vi"  # 替换为你的文件夹路径
output_csv = "vif_fusion.csv"  # 输出的 CSV 文件名
save_image_names_to_csv(folder_path, output_csv)
