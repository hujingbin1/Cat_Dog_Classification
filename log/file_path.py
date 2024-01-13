import os

# 指定要遍历的目录
directory = './'  # 请替换为实际的目录路径

# 用于存储文件名的列表
filenames = []

# 遍历目录
for root, dirs, files in os.walk(directory):
    for name in files:
        # 拼接完整的文件路径
        full_path = os.path.join(root, name)
        filenames.append(full_path)

# 将文件名保存到 TXT 文件中
output_file = 'file_list.txt'
with open(output_file, 'w') as file:
    for filename in filenames:
        file.write(filename + '\n')

print(f'文件名已保存到 {output_file}')
