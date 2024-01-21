import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader



def get_all_subdirectories(root_folder):
    subdirectories = []
    
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if dirpath != root_folder:
            subdirectories.append(os.path.abspath(dirpath))
        
        # 如果需要获取所有子目录的绝对路径，可以取消下一行的注释
        # subdirectories.extend([os.path.abspath(os.path.join(dirpath, subdir)) for subdir in dirnames])
    
    return subdirectories
    
    
#混杂因子实验时的dataset
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

# End2End模型的dataset,数据以[day,data]的形式存储，并且只有返回left_label,right_label
class End2EndCustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        #主文件夹下有几个子文件夹\主文件夹下有八个
        self.subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
        #类的数量
        self.classNum = len(self.subfolders)
        #print(self.classNum)
        self.file_paths = self.get_file_paths()
        #print(len(self.file_paths))
        #天数
        self.dayNum = len(self.file_paths)//self.classNum
        #print(self.dayNum)
        
    def __len__(self):
        total_length = 0
        for i in range(len(self.subfolders)):
            total_length += len(self.file_paths[0])
        return total_length

    def __getitem__(self, idx):
        left_labels = []
        right_labels = []
        datas = []
        index = idx
        start = 0
        for i in range(self.classNum):
            if (index-len(self.file_paths[i]))<0:
                index = index
                break
            else:
                index = index-len(self.file_paths[i])
            start += self.dayNum
                
        for i in range(self.dayNum): 
            file_path = self.file_paths[start+i][index]
            #print(file_path)
            # 从文件夹名称中提取s标签
            folder_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            #print(folder_name)
            left_label, right_label = map(float, folder_name.split('_'))
            left_labels.append(torch.tensor(left_label))
            right_labels.append(torch.tensor(right_label))
            data = np.loadtxt(file_path, delimiter=",", dtype=np.float32)
            # 在这里可以进行进一步的数据处理，例如转换为张量等
            data = torch.from_numpy(data).float()
            datas.append(data)
            # 返回数据和标签
        datas = torch.stack(datas)
        left_labels = torch.stack(left_labels)
        right_labels = torch.stack(right_labels)
        return datas, left_labels, right_labels
        
    def get_file_paths(self):
        file_paths = []
        for cur_dir in self.subfolders:
            #print(cur_dir)
            for sub_dir in get_all_subdirectories(cur_dir):
                path = []
                #print(sub_dir)
                for root, dirs, files in os.walk(sub_dir):
                    for file in files:
                        if file.endswith(".csv"):
                            path.append(os.path.join(root, file))
                file_paths.append(path)
        return file_paths
        
# 天数鲁棒性dataset,数据以[day,data]的形式存储，并且只有返回left_label,不需要right_label
class NewCustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        #主文件夹下有几个子文件夹\主文件夹下有0与1两个
        self.subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
        print(self.subfolders)
        self.file_paths = self.get_file_paths()

    def __len__(self):
        return len(self.file_paths[0])+len(self.file_paths[1])

    def __getitem__(self, idx):
        left_labels = []
        datas = []
        if idx<len(self.file_paths[0]):
            for i in range(len(self.file_paths)//2): 
                file_path = self.file_paths[i][idx]
                #print(file_path)
                # 从文件夹名称中提取s标签
                folder_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
                #print(folder_name)
                left_label, right_label = map(float, folder_name.split('_'))
                left_labels.append(torch.tensor(left_label))
                data = np.loadtxt(file_path, delimiter=",", dtype=np.float32)
                # 在这里可以进行进一步的数据处理，例如转换为张量等
                data = torch.from_numpy(data).float()
                datas.append(data)
            # 返回数据和标签
        else:
            for i in range(len(self.file_paths)//2): 
                file_path = self.file_paths[i+(len(self.file_paths)//2)][idx%(len(self.file_paths)//2)]
                #print(file_path)
                # 从文件夹名称中提取标签
                folder_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
                #print(folder_name)
                left_label, right_label = map(float, folder_name.split('_'))
                left_labels.append(torch.tensor(left_label))
                data = np.loadtxt(file_path, delimiter=",", dtype=np.float32)
                # 在这里可以进行进一步的数据处理，例如转换为张量等
                data = torch.from_numpy(data).float()
                datas.append(data)
        datas = torch.stack(datas)
        left_labels = torch.stack(left_labels)
        return datas, left_labels
        
    def get_file_paths(self):
        file_paths = []
        for cur_dir in self.subfolders:
            #print(cur_dir)
            for sub_dir in get_all_subdirectories(cur_dir):
                path = []
                #print(sub_dir)
                for root, dirs, files in os.walk(sub_dir):
                    for file in files:
                        if file.endswith(".csv"):
                            path.append(os.path.join(root, file))
                file_paths.append(path)
        return file_paths
        
# # 示例用法
# root_directory = "D:\\frequencyProcess\\end2end\\tr"
# dataset = End2EndCustomDataset(root_dir=root_directory)
# print(len(dataset))
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# # 遍历数据加载器
# for batch in dataloader:
    # data, left_label,right_label = batch
    # # 在这里处理每个批次的数据和标签
    # print(len(data))  # 假设你的数据是CSV格式，这里输出数据的形状
    # print(left_label)  # 输出左边和右边设备的标签
    # print(right_label)