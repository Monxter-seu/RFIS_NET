import os
import shutil
import random

#将source_folder中指定设备id+端口号的文件夹内指定数量的文件按照一定的比例分入训练集和测试集
#output_folder1是训练集，output_folder2是测试集
def extract_files(source_folder, device_id, port, output_folder1, output_folder2, file_num, ratio):
    # 构建port文件夹路径
    #print(source_folder)
    port_folder = os.path.join(source_folder, f"{device_id}_port{port}")
    #print(port_folder)
    #port_folder.encode('utf-8')
    #print(port_folder)

    # 检查port文件夹是否存在
    if not os.path.exists(port_folder):
        print(f"Error: Folder for port {port} does not exist.")
        return

    # 创建输出文件夹
    if not os.path.exists(output_folder1):
        os.makedirs(output_folder1)

    if not os.path.exists(output_folder2):
        os.makedirs(output_folder2)

    # 获取port文件夹下所有文件列表
    all_files = [f for f in os.listdir(port_folder) if f.endswith('.csv')]

    # 计算分配到两个文件夹的文件数量
    num_files_folder1 = file_num * ratio
    num_files_folder2 = file_num - num_files_folder1

    # 从随机选择的文件夹中复制文件到两个文件夹
    selected_files = random.sample(all_files, file_num)
    for i, file in enumerate(selected_files):
        if i < num_files_folder1:
            output_folder = output_folder1
        else:
            output_folder = output_folder2
        file_path = os.path.join(port_folder, file)
        #print(file_path)
        shutil.copy(file_path, output_folder)

    #print(f"Successfully extracted files from port {port} and distributed to two folders.")

# # 指定参数
# source_folder = 'D:\\csvProcessNew\\20240102'
# output_folder1 = 'D:\\csvProcessNew\\test1'
# output_folder2 = 'D:\\csvProcessNew\\test2'
# split_ratio = 0.7  # 指定分配的比例，这里是70%

# # 调用函数
# extract_files(source_folder,"pi06" , 1, output_folder1, output_folder2,1000, split_ratio)


#从source_folder下所有的日期文件夹中提取数据，划分为训练集测试集，随机选取，放入export_folder中
source_folder = 'D:\\csvProcessNew'
main_device_index = 2
split_ratio = 0.7
main_file_num = 1500
day_subfolders = [f.path for f in os.scandir(source_folder) if f.is_dir() and f.name != source_folder]
device_dic = ['pi06', 'rh01', 'rh02', 'rh03', 'tp47', 'tp49', 'tp50', 'wy00', 'wy01', 'wy02']
port_dic = [1,2,3,4]
vice_file_num = 1500 // (len(device_dic) - 1) 
export_folder = 'D:\\frequencyProcess\\DEVICE_RH02'

tr_dir = os.path.join(export_folder, 'tr')
tt_dir = os.path.join(export_folder, 'tt')

if not os.path.exists(export_folder):
    os.makedirs(export_folder)
    
if not os.path.exists(tr_dir):
    os.makedirs(tr_dir)

if not os.path.exists(tt_dir):
    os.makedirs(tt_dir)

for subfolder in day_subfolders:
    cur_tr_dir = os.path.join(tr_dir,os.path.basename(subfolder))
    cur_tt_dir = os.path.join(tt_dir,os.path.basename(subfolder))
    if not os.path.exists(cur_tr_dir):
        os.makedirs(cur_tr_dir)
    if not os.path.exists(cur_tt_dir):
        os.makedirs(cur_tt_dir)
    
    
    for device_id in device_dic:
        for port_index in port_dic:
            cur_tr_0_dir = os.path.join(cur_tr_dir,f"0_{port_index}")
            cur_tt_0_dir = os.path.join(cur_tt_dir,f"0_{port_index}")
            cur_tr_1_dir = os.path.join(cur_tr_dir,f"1_{port_index}")
            cur_tt_1_dir = os.path.join(cur_tt_dir,f"1_{port_index}")
            if not os.path.exists(cur_tr_0_dir):
                os.makedirs(cur_tr_0_dir)
            if not os.path.exists(cur_tt_0_dir):
                os.makedirs(cur_tt_0_dir)
            if not os.path.exists(cur_tr_1_dir):
                os.makedirs(cur_tr_1_dir)
            if not os.path.exists(cur_tt_1_dir):
                os.makedirs(cur_tt_1_dir)
                
            if device_id == device_dic[main_device_index]:
                extract_files(subfolder,device_id,port_index,cur_tr_0_dir,cur_tt_0_dir,main_file_num,split_ratio)
            else:
                extract_files(subfolder,device_id,port_index,cur_tr_1_dir,cur_tt_1_dir,vice_file_num,split_ratio)
