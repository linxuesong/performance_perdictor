import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# 定义自定义 Dataset 类
class ResourceDelayDataset(Dataset):
    def __init__(self, file_path):
        # 读取 Excel 文件（.xls 或 .xlsx）
        self.data = pd.read_excel(file_path, sheet_name="load_performance")
        # 特征（资源占比列）
        self.features = self.data[['SYolov5_cpu', 'SYolov5_mem', 'SYolov5_npu',
                                   'SBert_cpu', 'SBert_mem', 'SBert_npu',
                                   'SResnet_cpu', 'SResnet_mem', 'SResnet_npu']].values
        # 标签（延迟列）
        self.labels = self.data[['TYolov5_delay', 'TBert_delay', 'TResnet_delay']].values
        # 转换为张量
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(9, 64)  # 9 个输入特征（3 种服务 × 3 资源）
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)   # 输出 3 个延迟值

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化 Dataset 和 DataLoader
dataset = ResourceDelayDataset('/home/lxs/桌面/perdictor_data.xlsx')  # 替换为实际文件路径
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 初始化模型
model = MLP()

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练过程
num_epochs = 50  # 训练轮数
for epoch in range(num_epochs):
    model.train()  # 确保模型在训练模式
    running_loss = 0.0
    for batch_features, batch_labels in dataloader:
        # 前向传播
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # 打印每轮训练的平均损失
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}')

    # 保存每轮训练的模型（可选，可根据需求调整保存策略）
    # torch.save(model.state_dict(), f'model_epoch_{epoch + 1}.pth')

# 保存最终训练好的模型
torch.save(model.state_dict(), 'final_model.pth')

# ==================================
# 推理部分
# ==================================

# 加载训练好的模型
model.load_state_dict(torch.load('final_model.pth'))
model.eval()  # 切换到推理模式

# 准备测试数据（这里直接使用原始数据集中的一部分，实际应用中可替换为新数据）
test_features, test_labels = dataset[0:5]  # 取前5条数据测试

with torch.no_grad():  # 推理时不需要计算梯度
    predictions = model(test_features)
    print("预测延迟值：")
    print(predictions)
    print("真实延迟值：")
    print(test_labels)
