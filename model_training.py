#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2024/6/25 12:46
# @Author: william.cao
# @File  : model_training.py.py
# 模型初始化和训练部分
import json
import os
import pandas as pd
import torch
from torch._dynamo.config import base_dir
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import logging

# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 全局变量
model = None
tokenizer = None

def load_config():
    with open('config.json', 'r', encoding='utf-8') as file:
        config = json.load(file)
    return config

def save_config(config):
    with open('config.json', 'w', encoding='utf-8') as file:
        json.dump(config, file, ensure_ascii=False, indent=4)

class LogDataset(Dataset):
    """
    自定义数据集类，用于处理日志数据并转换为模型输入格式
    """

    def __init__(self, csv_file, tokenizer, max_length=512):
        """
        初始化 LogDataset 数据集类
        :param csv_file: 包含日志数据的 CSV 文件路径
        :param tokenizer: 用于文本编码的 CodeBERT 分词器
        :param max_length: 最大序列长度，默认为 512
        """
        self.data = pd.read_csv(csv_file)  # 读取 CSV 文件中的数据
        if self.data.empty:
            raise ValueError(f"CSV 文件 {csv_file} 是空的或不存在.")
        self.tokenizer = tokenizer  # CodeBERT 分词器
        self.max_length = max_length  # 最大序列长度

    def __len__(self):
        """
        返回数据集的大小（样本数量）
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        根据索引获取数据集中的一个样本
        :param idx: 样本索引
        :return: 包含输入 IDs、注意力掩码和标签的字典
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()  # 如果索引是张量，则转换为列表

        log_message = self.data.iloc[idx, 0]  # 获取日志消息
        label = torch.tensor(self.data.iloc[idx, 1])  # 获取对应的标签，并转换为张量

        # 使用分词器对日志消息进行编码
        encoding = self.tokenizer(
            log_message,
            max_length=self.max_length,  # 设置最大长度
            padding='max_length',  # 使用最大长度进行填充
            truncation=True,  # 进行截断
            return_tensors='pt'  # 返回 PyTorch 张量
        )

        # 移除多余的维度（从形状 [1, max_length] 转换为 [max_length]）
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # 构建样本字典
        sample = {
            'input_ids': input_ids,  # 输入 IDs
            'attention_mask': attention_mask,  # 注意力掩码
            'label': label  # 标签
        }

        return sample  # 返回样本


def initialize_model_and_tokenizer():
    global model, tokenizer
    if model is not None and tokenizer is not None:
        return model, tokenizer
    try:
        model_path = "E:\\Bert-Log-analysis-monitoring\\Bert-Log-analysis-monitoring\\codebert-base"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 更安全地加载模型和 tokenizer
        model = RobertaForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            torch_dtype=torch.float32,
            use_safetensors=True,
            low_cpu_mem_usage=False  # ✅ 禁用懒加载，避免 meta tensor
        )
        tokenizer = RobertaTokenizer.from_pretrained(model_path)

        model = model.to(device)

        logging.info(f"模型和分词器初始化成功，运行在 {device} 上")

        # 检查是否有 meta tensor（用于断言懒加载是否还存在）
        for name, param in model.named_parameters():
            if param.device.type == 'meta':
                raise RuntimeError(f"模型参数 {name} 是 meta tensor，说明模型未正确加载")

        return model, tokenizer
    except Exception as e:
        logging.error(f"模型和分词器初始化失败: {e}")
        return None, None


class FocalLoss(torch.nn.Module):
    """
    定义 Focal Loss 损失函数，用于解决类别不平衡问题
    """
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()

async def train_model(csv_file='logs.csv', num_epochs=3, validation_split=0.1, patience=3):
    """
    训练 CodeBERT 模型并保存
    :param csv_file: 包含日志数据的 CSV 文件路径
    :param num_epochs: 训练轮数
    :param validation_split: 验证集比例
    :param patience: 早停机制容忍度
    """
    global model, tokenizer
    model, tokenizer = initialize_model_and_tokenizer()
    if model is None or tokenizer is None:
        logging.error("模型和分词器初始化失败，训练终止")
        return

    dataset = LogDataset(csv_file, tokenizer)

    # 计算类别权重
    class_counts = dataset.data.iloc[:, 1].value_counts()
    alpha = torch.tensor([class_counts[1] / len(dataset.data), class_counts[0] / len(dataset.data)])

    # 分割数据集为训练集和验证集
    train_size = int((1 - validation_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    if train_size == 0 or val_size == 0:
        logging.error("训练集或验证集的样本数为零，训练终止")
        return

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # 初始化优化器
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # 设置学习率调度器
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # 检查是否有GPU可用
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # 使用 Focal Loss
    criterion = FocalLoss(gamma=2, alpha=alpha.to(device))

    best_val_loss = float('inf')
    epochs_no_improve = 0

    # 训练循环
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        total_train_loss = 0

        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()  # 清空梯度
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = criterion(outputs.logits, labels)
            total_train_loss += loss.item()

            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            scheduler.step()  # 更新学习率

        avg_train_loss = total_train_loss / len(train_dataloader)
        logging.info(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss}')

        model.eval()  # 设置模型为评估模式
        total_val_loss = 0

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = criterion(outputs.logits, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        logging.info(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            model_save_path = os.path.join(base_dir, 'Bert-Log-analysis-monitoring/codebert-base')
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            torch.save(model.state_dict(), os.path.join(model_save_path, 'model_state_dict.pth'))
            logging.info("验证集损失降低，保存最佳模型")
        else:
            epochs_no_improve += 1
            logging.info(f'验证集损失未降低次数：{epochs_no_improve}')

        if epochs_no_improve >= patience:
            logging.info(f'验证集损失连续 {patience} 次未降低，提前停止训练')
            break

    model_save_path = os.path.join(base_dir, 'Bert-Log-analysis-monitoring/codebert-base')
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    torch.save(model.state_dict(), os.path.join(model_save_path, 'model_state_dict.pth'))
    logging.info("最终模型保存成功")
