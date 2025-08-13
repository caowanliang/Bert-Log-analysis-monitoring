#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2024/6/25 12:45
# @Author: william.cao
# @File  : data_preparation.py.py
# 数据准备
import hashlib
import os
import csv
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def store_to_csv(message, label, file_path):
    """
    将日志信息和标签存储到 CSV 文件中。
    :param message: 日志消息
    :param label: 日志标签
    :param file_path: CSV 文件路径
    """
    try:
        file_exists = os.path.isfile(file_path)
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['message', 'label'])
            writer.writerow([message, label])
    except Exception as e:
        logging.error(f"存储到 CSV 文件时出错: {e}")


def store_to_parquet(message, label, file_path):
    """存储为Parquet格式"""
    try:
        df = pd.DataFrame({'message': [message], 'label': [label]})
        if os.path.exists(file_path):
            existing_df = pd.read_parquet(file_path)
            df = pd.concat([existing_df, df])
        df.to_parquet(file_path, index=False)
    except Exception as e:
        logging.error(f"存储到Parquet文件时出错: {e}")
