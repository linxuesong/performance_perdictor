import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

# 宏定义
TASK_TYPES = ["Yolov5_B", "Yolov5_M", "Yolov5_S"]
SERVICE_TYPES = ["Yolov5"]
RESOURCE_TYPES = ["cpu", "mem", "npu"]
# Sensitivity表里面压力点个数
SAMPLING_POINT_CNTS = 10
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
    "SOLE_CPU": "cpu",
    "SOLE_MEM": "mem",
    "SOLE_NPU": "npu",
    "SOLE_DELAY": "delay",

    # competition_intensity表列名
    "COMPETITION_DEV_TYPE": "dev_type",
    "COMPETITION_TASK_TYPE": "task_type",
    "COMPETITION_CPU": "cpu",
    "COMPETITION_MEM": "mem",
    "COMPETITION_NPU": "npu",

    # load_performance表列名
    "LOAD_DEV_TYPE": "dev_type",
    "LOAD_TOTAL_CPU": "total_cpu",
    "LOAD_TOTAL_MEM": "total_mem",
    "LOAD_TOTAL_NPU": "total_npu",
    "LOAD_NEW_TASK":  "new_task",
    "LOAD_CUR_TASKS": "current_tasks",
    "LOAD_DESCEND_PERFORMANCE": "descend_performance"
}

# 从 Excel 文件读取敏感度曲线数据
def read_sensitivity_data(file_path):
    df = pd.read_excel(file_path, sheet_name=EXCEL_STRUCTURE["SENSITIVITY_SHEET"])
    sensitivity_info = {}
    for _, row in df.iterrows():
        dev_type = row[EXCEL_STRUCTURE["SENSITIVITY_DEV_TYPE"]]
        task_type = row[EXCEL_STRUCTURE["SENSITIVITY_TASK_TYPE"]]
        if dev_type not in sensitivity_info:
            sensitivity_info[dev_type] = {}
        if task_type not in sensitivity_info[dev_type]:
            sensitivity_info[dev_type][task_type] = {}

        resource_info = {}
        for resource in RESOURCE_TYPES:
            pressure_key = f"{resource.upper()}_PRESSURE"
            delay_key = f"{resource.upper()}_DELAY"
            resource_info[resource] = {
                'pressure': eval(row[EXCEL_STRUCTURE[pressure_key]]),
                'delay': eval(row[EXCEL_STRUCTURE[delay_key]])
            }

        sensitivity_info[dev_type][task_type] = resource_info
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

        resource_info = {}
        for resource in RESOURCE_TYPES:
            key = f"SOLE_{resource.upper()}"
            resource_info[resource] = row[EXCEL_STRUCTURE[key]]

        resource_info["delay"] = delay
        solo_info[dev_type][task_type] = resource_info
    return solo_info

# 从 Excel 文件读取负载性能数据
def read_load_performance_data(file_path):
    df = pd.read_excel(file_path, sheet_name=EXCEL_STRUCTURE["LOAD_PERFORMANCE_SHEET"])
    load_performance_data = []
    for _, row in df.iterrows():
        data = {
            "dev_type": row[EXCEL_STRUCTURE["LOAD_DEV_TYPE"]],
            "new_task": row[EXCEL_STRUCTURE["LOAD_NEW_TASK"]],
            "cur_task_type": eval(row["cur_task_type"]),
            "descend_performance": row[EXCEL_STRUCTURE["LOAD_DESCEND_PERFORMANCE"]]
        }
        for resource in RESOURCE_TYPES:
            total_key = f"LOAD_TOTAL_{resource.upper()}"
            data[f"total_{resource}"] = row[EXCEL_STRUCTURE[total_key]]

        load_performance_data.append(data)
    return load_performance_data

# 统一计算函数
def calculate_stats(cur_tasks, dev_type, solo_info):
    old_task_sum = 0
    resource_sums = {resource: 0 for resource in RESOURCE_TYPES}
    resource_means = {resource: 0 for resource in RESOURCE_TYPES}
    resource_variances = {resource: 0 for resource in RESOURCE_TYPES}

    # 计算任务总数和资源总和
    for task_id, num in enumerate(cur_tasks):
        if num > 0:
            old_task_sum += num
            old_task_type = TASK_TYPES[task_id]
            for resource in RESOURCE_TYPES:
                resource_sums[resource] += solo_info[dev_type][old_task_type][resource] * num

    # 计算资源平均值
    if old_task_sum > 0:
        for resource in RESOURCE_TYPES:
            resource_means[resource] = resource_sums[resource] / old_task_sum

    # 计算资源方差
    for task_id, num in enumerate(cur_tasks):
        if num > 0:
            old_task_type = TASK_TYPES[task_id]
            for resource in RESOURCE_TYPES:
                value = solo_info[dev_type][old_task_type][resource]
                resource_variances[resource] += (value - resource_means[resource]) ** 2 * num

    if old_task_sum > 0:
        for resource in RESOURCE_TYPES:
            resource_variances[resource] /= old_task_sum

    # 组装结果
    result = [old_task_sum]
    for resource in RESOURCE_TYPES:
        result.extend([resource_means[resource], resource_variances[resource]])

    return result

# 特征工程与数据集类
class LoadDataset(Dataset):
    def __init__(self, load_performance_data, sensitivity_info, solo_info):
        self.load_performance_data = load_performance_data
        self.sensitivity_info = sensitivity_info
        self.solo_info = solo_info

    def __len__(self):
        return len(self.load_performance_data)

    def __getitem__(self, idx):
        load_performance_row = self.load_performance_data[idx]
        dev_type = load_performance_row['dev_type']
        new_task_type = load_performance_row['new_task']

        # 敏感度曲线特征 pressure[0,1,0.2,0.3] delay[10,12,15] 这里stack按照轴1堆叠成 [[0.1, 10], [0.2, 12], [0.3,15]]
        const_data = self.sensitivity_info[dev_type][new_task_type]
        cpu_seq = np.stack([const_data['cpu']['pressure'], const_data['cpu']['delay']], axis=1)
        mem_seq = np.stack([const_data['mem']['pressure'], const_data['mem']['delay']], axis=1)
        npu_seq = np.stack([const_data['npu']['pressure'], const_data['npu']['delay']], axis=1)

        # 混部压力信息 不同任务[任务a数量,cpu,mem,npu]-》 任务总量,cpu平均，cpu方差
        cur_tasks = load_performance_row['cur_task_type']

        competition_factor = calculate_stats(cur_tasks, dev_type, self.solo_info)

        # 新任务延迟的变化
        descend_performance = load_performance_row['descend_performance']

        return (
            torch.FloatTensor(cpu_seq),
            torch.FloatTensor(mem_seq),
            torch.FloatTensor(npu_seq),
            torch.FloatTensor(competition_factor),
            torch.FloatTensor([descend_performance])
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

        # 竞争因子特征处理
        self.competition_net = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU()
        ).to(device)

        # 融合特征处理
        self.fusion_net = nn.Sequential(
            nn.Linear(64 + 32 * len(RESOURCE_TYPES), 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        ).to(device)

    def forward(self, cpu_seq, mem_seq, npu_seq, competition_factor):
        # 资源曲线特征
        cpu_feat = self.lstm(cpu_seq)
        mem_feat = self.lstm(mem_seq)
        npu_feat = self.lstm(npu_seq)

        # 竞争因子特征
        competition_feat = self.competition_net(competition_factor)

        # 特征融合
        combined = torch.cat([
            competition_feat,
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
            for cpu_seq, mem_seq, npu_seq, competition_factor, true_delay in train_loader:
                cpu_seq = cpu_seq.to(self.device)
                mem_seq = mem_seq.to(self.device)
                npu_seq = npu_seq.to(self.device)
                competition_factor = competition_factor.to(self.device)
                true_delay = true_delay.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(cpu_seq, mem_seq, npu_seq, competition_factor)
                loss = criterion(outputs, true_delay)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * cpu_seq.size(0)

            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for cpu_seq, mem_seq, npu_seq, competition_factor, true_delay in val_loader:
                    cpu_seq = cpu_seq.to(self.device)
                    mem_seq = mem_seq.to(self.device)
                    npu_seq = npu_seq.to(self.device)
                    competition_factor = competition_factor.to(self.device)
                    true_delay = true_delay.to(self.device)

                    outputs = self.model(cpu_seq, mem_seq, npu_seq, competition_factor)
                    val_loss += criterion(outputs, true_delay).item() * cpu_seq.size(0)

            # 打印训练进度
            train_loss /= len(train_dataset)
            val_loss /= len(val_dataset)
            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# 推理函数
def inference(params, sensitivity_info, solo_info, model, device):
    print("输入参数 params:", params)
    # 提取敏感度曲线
    dev_type = params['dev_type']
    task_type = params['new_task']
    const_data = sensitivity_info[dev_type][task_type]
    cpu_seq = np.stack([const_data['cpu']['pressure'], const_data['cpu']['delay']], axis=1)
    mem_seq = np.stack([const_data['mem']['pressure'], const_data['mem']['delay']], axis=1)
    npu_seq = np.stack([const_data['npu']['pressure'], const_data['npu']['delay']], axis=1)

    # 计算 competition_factor
    cur_task = params['cur_task_type']
    competition_factor = calculate_stats(cur_task, dev_type, solo_info)
    print("competition_factor:", competition_factor)

    # 转换为张量
    cpu_seq = torch.FloatTensor(cpu_seq).unsqueeze(0).to(device)
    mem_seq = torch.FloatTensor(mem_seq).unsqueeze(0).to(device)
    npu_seq = torch.FloatTensor(npu_seq).unsqueeze(0).to(device)
    competition_factor = torch.FloatTensor(competition_factor).unsqueeze(0).to(device)

    # 执行预测
    with torch.no_grad():
        prediction = model(cpu_seq, mem_seq, npu_seq, competition_factor)

    result = prediction.item()
    print("最后返回结果:", result)
    return result

if __name__ == "__main__":
    file_path = r"D:\work\myproject\performance_perdictor\base_task_infulence_predictor\task_influence_perdictor_data.xlsx"

    # 读取数据
    sensitivity_info = read_sensitivity_data(file_path)
    solo_info = read_sole_performance_data(file_path)
    load_performance_data = read_load_performance_data(file_path)

    # 构建数据集
    dataset = LoadDataset(load_performance_data, sensitivity_info, solo_info)

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
        'new_task': 'Yolov5_B',
        'cur_task_type': [1, 2, 3]
    }

    predicted_delay = inference(test_params, sensitivity_info, solo_info, model, device)
    print(f"预测延迟: {predicted_delay:.2f} ms")    