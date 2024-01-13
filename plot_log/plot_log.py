import matplotlib.pyplot as plt
import re
import os
import csv

# 读取 file_list.txt 文件
with open('file_list.txt', 'r') as file:
    paths = file.readlines()
    paths = [path.strip() for path in paths]  # 去除每行末尾的换行符

for log_file_path in paths:
    # 提取子文件夹名称（例如：resnet50_lr0.01_opt-adam_bs64）
    path_parts = log_file_path.split(os.sep)  # 使用 os.sep 保证跨平台兼容性
    subfolder_name = path_parts[-2]  # 倒数第二部分为子文件夹名称

    # 读取日志文件
    with open(log_file_path, 'r') as file:
        log_data = file.readlines()

    # 解析数据
    train_losses = []
    val_accs = []

    for line in log_data:
        train_loss_match = re.search(r'train Loss: ([\d.eE+-]+)', line)
        val_acc_match = re.search(r'val Acc: ([\d.eE+-]+)', line)

        if train_loss_match:
            train_losses.append(float(train_loss_match.group(1)))

        if val_acc_match:
            val_accs.append(float(val_acc_match.group(1)))

    # 保存数据到 CSV 文件
    csv_folder = 'csv_data'
    subfolder_path = os.path.join(csv_folder, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)
    csv_path = os.path.join(subfolder_path, 'training_data.csv')

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Train Loss', 'Validation Accuracy'])
        for i in range(len(train_losses)):
            writer.writerow([i + 1, train_losses[i], val_accs[i] if i < len(val_accs) else 'NA'])

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    # 创建 img 文件夹并保存图像
    img_folder = 'img'
    subfolder_path = os.path.join(img_folder, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)
    img_path = os.path.join(subfolder_path, 'training_validation_plot.png')
    plt.savefig(img_path)
    plt.close()  # 关闭图表以释放内存

    print(f"Image saved at: {img_path}")
    print(f"CSV data saved at: {csv_path}")
