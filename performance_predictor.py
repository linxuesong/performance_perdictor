import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

# 宏定义
TASK_TYPES = ["Yolov5", "Bert", "Resnet"]
SERVICE_TYPES = ["Yolov5", "Bert", "Resnet"]
RESOURCE_TYPES = ["cpu", "mem", "npu"]

# 静态全局变量管理Excel表格结构
EXCEL_STRUCTURE = {
    # 工作表名
    "SENSITIVITY_SHEET": "sensitivity",
    "SOLE_PERFORMANCE_SHEET": "sole_performance",
    "LOAD_PERFORMANCE_SHEET": "load_performance",

    # Sensitivity表列名
    "SENSITIVITY_DEV_TYPE": "dev_type",
    "SENSITIVITY_TASK_TYPE": "task_type",
    "CPU_PRESSURE": "cpu_pressure_sequence",
    "CPU_DELAY": "cpu_delay_sequence",
    "MEM_PRESSURE": "mem_pressure_sequence",
    "MEM_DELAY": "mem_delay_sequence",
    "NPU_PRESSURE": "npu_pressure_sequence",
    "NPU_DELAY": "npu_delay_sequence",

    # solo_performance表列名
    "SOLE_DEV_TYPE": "dev_type",
    "SOLE_TASK_TYPE": "task_type",
    "SOLE_DELAY": "delay"
     
}

# 从 Excel 文件读取敏感度曲线数据
def read_sensitivity_data(file_path):
    df = pd.read_excel(file_path, sheet_name=EXCEL_STRUCTURE["SENSITIVITY_SHEET"])
    sensitivity_info = {}
    for _, row in df.iterrows():
        dev_type = row[EXCEL_STRUCTURE["SENSITIVITY_DEV_TYPE"]]
        task_type = row[EXCEL_STRUCTURE["SENSITIVITY_TASK_TYPE"]]
        if task_type not in sensitivity_info:
            sensitivity_info[task_type] = {}
        if dev_type not in sensitivity_info[task_type]:
            sensitivity_info[task_type][dev_type] = {}

        sensitivity_info[task_type][dev_type]['sensitivity'] = {
            'cpu': {
                'pressure': eval(row[EXCEL_STRUCTURE["CPU_PRESSURE"]]),
                'delay': eval(row[EXCEL_STRUCTURE["CPU_DELAY"]])
            },
            'mem': {
                'pressure': eval(row[EXCEL_STRUCTURE["MEM_PRESSURE"]]),
                'delay': eval(row[EXCEL_STRUCTURE["MEM_DELAY"]])
            },
            'npu': {
                'pressure': eval(row[EXCEL_STRUCTURE["NPU_PRESSURE"]]),
                'delay': eval(row[EXCEL_STRUCTURE["NPU_DELAY"]])
            }
        }
    return sensitivity_info

# 从 Excel 文件读取单独任务延迟
def read_sole_performance_data(file_path):
    df = pd.read_excel(file_path, sheet_name=EXCEL_STRUCTURE["SOLE_PERFORMANCE_SHEET"])
    solo_info = {}
    for _, row in df.iterrows():
        dev_type = row[EXCEL_STRUCTURE["SOLE_DEV_TYPE"]]
        task_type = row[EXCEL_STRUCTURE["SOLE_TASK_TYPE"]]
        delay = row[EXCEL_STRUCTURE["SOLE_DELAY"]]
        if dev_type not in solo_info:
            solo_info[dev_type] = {}
        solo_info[dev_type][task_type] = delay
    return solo_info

# 从 Excel 文件读取负载数据并重组
def read_load_performance_data(file_path):
    df = pd.read_excel(file_path, sheet_name=EXCEL_STRUCTURE["LOAD_PERFORMANCE_SHEET"])
    new_data = []
    for _, row in df.iterrows():
        dev_type = row['dev_type']
        # 处理Yolov5任务
        yolov5_data = {
            'dev_type': dev_type,
            'task_type': 'Yolov5',
            'SYolov5_cpu': row['SYolov5_cpu'],
            'SYolov5_mem': row['SYolov5_mem'],
            'SYolov5_npu': row['SYolov5_npu'],
            'SBert_cpu': row['SBert_cpu'],
            'SBert_mem': row['SBert_mem'],
            'SBert_npu': row['SBert_npu'],
            'SResnet_cpu': row['SResnet_cpu'],
            'SResnet_mem': row['SResnet_mem'],
            'SResnet_npu': row['SResnet_npu'],
            'delay': row['TYolov5_delay']
        }
        new_data.append(yolov5_data)
        # 处理Bert任务
        bert_data = {
            'dev_type': dev_type,
            'task_type': 'Bert',
            'SYolov5_cpu': row['SYolov5_cpu'],
            'SYolov5_mem': row['SYolov5_mem'],
            'SYolov5_npu': row['SYolov5_npu'],
            'SBert_cpu': row['SBert_cpu'],
            'SBert_mem': row['SBert_mem'],
            'SBert_npu': row['SBert_npu'],
            'SResnet_cpu': row['SResnet_cpu'],
            'SResnet_mem': row['SResnet_mem'],
            'SResnet_npu': row['SResnet_npu'],
            'delay': row['TBert_delay']
        }
        new_data.append(bert_data)
        # 处理Resnet任务
        resnet_data = {
            'dev_type': dev_type,
            'task_type': 'Resnet',
            'SYolov5_cpu': row['SYolov5_cpu'],
            'SYolov5_mem': row['SYolov5_mem'],
            'SYolov5_npu': row['SYolov5_npu'],
            'SBert_cpu': row['SBert_cpu'],
            'SBert_mem': row['SBert_mem'],
            'SBert_npu': row['SBert_npu'],
            'SResnet_cpu': row['SResnet_cpu'],
            'SResnet_mem': row['SResnet_mem'],
            'SResnet_npu': row['SResnet_npu'],
            'delay': row['TResnet_delay']
        }
        new_data.append(resnet_data)
    return new_data

# 特征工程与数据集类
class LoadDataset(Dataset):
    def __init__(self, load_data, sensitivity_info, solo_info):
        self.load_data = load_data
        self.sensitivity_info = sensitivity_info
        self.solo_info = solo_info

    def __len__(self):
        return len(self.load_data)

    def __getitem__(self, idx):
        row = self.load_data[idx]
        dev_type = row['dev_type']
        task_type = row['task_type']
        
        # 服务负载特征
        service_load = [
            row['SYolov5_cpu'], row['SYolov5_mem'], row['SYolov5_npu'],
            row['SBert_cpu'], row['SBert_mem'], row['SBert_npu'],
            row['SResnet_cpu'], row['SResnet_mem'], row['SResnet_npu']
        ]
        
        # 敏感度曲线特征
        const_data = self.sensitivity_info[task_type][dev_type]['sensitivity']
        cpu_seq = np.stack([const_data['cpu']['pressure'], const_data['cpu']['delay']], axis=1)
        mem_seq = np.stack([const_data['mem']['pressure'], const_data['mem']['delay']], axis=1)
        npu_seq = np.stack([const_data['npu']['pressure'], const_data['npu']['delay']], axis=1)
        
        # Solo性能特征
        solo_delay = self.solo_info[dev_type][task_type]
        
        # 目标延迟
        true_delay = row['delay']
        
        return (
            torch.FloatTensor(service_load),
            torch.FloatTensor([solo_delay]),
            torch.FloatTensor(cpu_seq),
            torch.FloatTensor(mem_seq),
            torch.FloatTensor(npu_seq),
            torch.FloatTensor([true_delay])
        )

# 处理各资源敏感度曲线的 LSTM
class ResourceLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out[:, -1, :]

# 延迟预测模型
class DelayPredictor(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.lstm = ResourceLSTM().to(device)
        
        # 服务负载特征处理
        self.service_net = nn.Sequential(
            nn.Linear(len(SERVICE_TYPES)*len(RESOURCE_TYPES), 64),
            nn.ReLU()
        ).to(device)
        
        # Solo性能特征处理
        self.solo_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU()
        ).to(device)
        
        # 融合特征处理
        self.fusion_net = nn.Sequential(
            nn.Linear(64 + 32 + 32*len(RESOURCE_TYPES), 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        ).to(device)

    def forward(self, service_load, solo_delay, cpu_seq, mem_seq, npu_seq):
        # 服务负载特征
        service_feat = self.service_net(service_load)
        
        # Solo性能特征
        solo_feat = self.solo_net(solo_delay)
        
        # 资源曲线特征
        cpu_feat = self.lstm(cpu_seq)
        mem_feat = self.lstm(mem_seq)
        npu_feat = self.lstm(npu_seq)
        
        # 特征融合
        combined = torch.cat([
            service_feat,
            solo_feat,
            cpu_feat,
            mem_feat,
            npu_feat
        ], dim=1)
        
        return self.fusion_net(combined)

# 训练系统
class TrainingSystem:
    def __init__(self, device):
        self.device = device
        self.model = DelayPredictor(device)
        self.model.to(device)

    def train(self, train_dataset, val_dataset, epochs=100, batch_size=32):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 2)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for service_load, solo_delay, cpu_seq, mem_seq, npu_seq, true_delay in train_loader:
                service_load = service_load.to(self.device)
                solo_delay = solo_delay.to(self.device)
                cpu_seq = cpu_seq.to(self.device)
                mem_seq = mem_seq.to(self.device)
                npu_seq = npu_seq.to(self.device)
                true_delay = true_delay.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(service_load, solo_delay, cpu_seq, mem_seq, npu_seq)
                loss = criterion(outputs, true_delay.unsqueeze(1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * service_load.size(0)

            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for service_load, solo_delay, cpu_seq, mem_seq, npu_seq, true_delay in val_loader:
                    service_load = service_load.to(self.device)
                    solo_delay = solo_delay.to(self.device)
                    cpu_seq = cpu_seq.to(self.device)
                    mem_seq = mem_seq.to(self.device)
                    npu_seq = npu_seq.to(self.device)
                    true_delay = true_delay.to(self.device)

                    outputs = self.model(service_load, solo_delay, cpu_seq, mem_seq, npu_seq)
                    val_loss += criterion(outputs, true_delay.unsqueeze(1)).item() * service_load.size(0)

            # 打印训练进度
            train_loss /= len(train_dataset)
            val_loss /= len(val_dataset)
            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# 推理函数
def inference(params, sensitivity_info, solo_info, model, device):
    # 构造输入数据
    service_load = [
        params['SYolov5_cpu'], params['SYolov5_mem'], params['SYolov5_npu'],
        params['SBert_cpu'], params['SBert_mem'], params['SBert_npu'],
        params['SResnet_cpu'], params['SResnet_mem'], params['SResnet_npu']
    ]
    
    # 提取敏感度曲线
    const_data = sensitivity_info[params['task_type']][params['dev_type']]['sensitivity']
    cpu_seq = np.stack([const_data['cpu']['pressure'], const_data['cpu']['delay']], axis=1)
    mem_seq = np.stack([const_data['mem']['pressure'], const_data['mem']['delay']], axis=1)
    npu_seq = np.stack([const_data['npu']['pressure'], const_data['npu']['delay']], axis=1)
    
    # 转换为张量
    service_load = torch.FloatTensor(service_load).unsqueeze(0).to(device)
    solo_delay = torch.FloatTensor([solo_info[params['dev_type']][params['task_type']]]).unsqueeze(0).to(device)
    cpu_seq = torch.FloatTensor(cpu_seq).unsqueeze(0).to(device)
    mem_seq = torch.FloatTensor(mem_seq).unsqueeze(0).to(device)
    npu_seq = torch.FloatTensor(npu_seq).unsqueeze(0).to(device)
    
    # 执行预测
    with torch.no_grad():
        prediction = model(service_load, solo_delay, cpu_seq, mem_seq, npu_seq)
    return prediction.item()

if __name__ == "__main__":
    file_path = "/home/lxs/桌面/perdictor_data.xlsx"
    
    # 读取数据
    sensitivity_info = read_sensitivity_data(file_path)
    solo_info = read_sole_performance_data(file_path)
    load_data = read_load_performance_data(file_path)
    
    # 构建数据集
    dataset = LoadDataset(load_data, sensitivity_info, solo_info)
    
    # 划分训练集和验证集
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # 初始化训练系统并训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_system = TrainingSystem(device)
    training_system.train(train_dataset, val_dataset, epochs=200)
    
    # 保存模型（示例）
    torch.save(training_system.model.state_dict(), "delay_model.pth")
    
    # 推理示例
    model = DelayPredictor(device)
    model.load_state_dict(torch.load("delay_model.pth"))
    model.eval()
    
    test_params = {
        'dev_type': 'ATLAS',
        'task_type': 'Yolov5',
        'SYolov5_cpu': 0.4,
        'SYolov5_mem': 0.3,
        'SYolov5_npu': 0.6,
        'SBert_cpu': 0.15,
        'SBert_mem': 0.1,
        'SBert_npu': 0.25,
        'SResnet_cpu': 0.0,
        'SResnet_mem': 0.0,
        'SResnet_npu': 0.0
    }
    
    predicted_delay = inference(test_params, sensitivity_info, solo_info, model, device)
    print(f"预测延迟: {predicted_delay:.2f} ms")
