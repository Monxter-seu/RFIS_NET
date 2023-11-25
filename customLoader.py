import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_paths = self.get_file_paths()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        # 从文件夹名称中提取标签
        folder_name = os.path.basename(os.path.dirname(file_path))
        left_label, right_label = map(float, folder_name.split('_'))
        
        data = np.loadtxt(file_path, delimiter=",", dtype=np.float32)  # 假设你的数据是CSV格式的
        # 在这里可以进行进一步的数据处理，例如转换为张量等
        data = torch.from_numpy(data).float()
        # 返回数据和标签
        return data, left_label, right_label

    def get_file_paths(self):
        file_paths = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".csv"):
                    file_paths.append(os.path.join(root, file))
        return file_paths

# # 示例用法
# root_directory = "D:\\frequencyProcess\\test\\tr\\mix"
# dataset = CustomDataset(root_dir=root_directory)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# # 遍历数据加载器
# for batch in dataloader:
    # data, left_label, right_label = batch
    # # 在这里处理每个批次的数据和标签
    # print(data.shape)  # 假设你的数据是CSV格式，这里输出数据的形状
    # print(left_label, right_label)  # 输出左边和右边设备的标签