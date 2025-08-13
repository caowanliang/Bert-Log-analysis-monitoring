#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2024/6/25 12:48
# @Author: william.cao
# @File  : error_analysis.py
# bert错误日志分析
import os
import re
import torch
import logging

from model_training import initialize_model_and_tokenizer, load_config
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 初始化模型和分词器，只执行一次
model, tokenizer = None, None

def preprocess_log_message(log_message):
    """
    预处理日志消息，去除无关信息并提取有用特征。
    :param log_message: 原始日志消息字符串
    :return: 预处理后的日志消息字符串
    """
    if log_message is None:
        return log_message

    # 去除时间戳（包括毫秒部分）
    log_message = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{3})?', '', log_message)
    # 去除 IP 地址
    log_message = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '', log_message)
    # 去除中括号内容
    log_message = re.sub(r'\[.*?\]', '', log_message)
    # 去除 ANSI 转义序列
    log_message = re.sub(r'\x1B[@-_][0-?]*[ -/]*[@-~]', '', log_message)
    # 去除多余的转义字符（如 ）
    log_message = re.sub(r'\x1b', '', log_message)
    # 去除所有数字
    log_message = re.sub(r'\d+', '', log_message)

    return log_message.strip()


async def analyze_log_with_codebert(log_message):
    """使用CodeBERT模型分析日志消息"""
    try:
        model_path = "E:\\Bert-Log-analysis-monitoring\\Bert-Log-analysis-monitoring\\codebert-base"
        
        # 1. 先加载模型到CPU
        model = RobertaForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            torch_dtype=torch.float32  # 明确指定数据类型
        )
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        
        # 2. 检查设备并正确转移模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if device.type == 'cuda':
            # 对于GPU使用to_empty()方法
            model = model.to_empty(device=device)
            # 重新加载模型权重
            state_dict = torch.load(
                os.path.join(model_path, 'pytorch_model.bin'),
                map_location=device
            )
            model.load_state_dict(state_dict)
        else:
            # CPU直接转移
            model = model.to(device)
            
        # 3. 执行推理
        inputs = tokenizer(log_message, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            confidence = torch.sigmoid(logits).item()
            
        return confidence
        
    except Exception as e:
        logging.error(f"CodeBERT分析失败: {e}")
        return None
