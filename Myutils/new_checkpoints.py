import os
import shutil

def manage_checkpoints(base_dir=".",enhancement=10, max_checkpoints=10):
    # 获取所有符合 "checkpoint-*" 格式的目录
    checkpoints = [d for d in os.listdir(base_dir) if d.startswith("checkpoint-") and d[11:].isdigit()]

    # 按编号排序
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))

    # 如果超过 max_checkpoints，则删除最早的
    if len(checkpoints) >= max_checkpoints:
        oldest_checkpoint = checkpoints.pop(0)  # 取出最早的 checkpoint
        shutil.rmtree(os.path.join(base_dir, oldest_checkpoint))  # 删除文件夹

    # # 计算新的 checkpoint 号
    # new_index = int(checkpoints[-1].split("-")[1]) + enhancement if checkpoints else 1
    # new_checkpoint =f"checkpoint-{new_index}"
    # return new_checkpoint  # 返回新创建的 checkpoint 目录