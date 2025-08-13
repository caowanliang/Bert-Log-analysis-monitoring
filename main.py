#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2024/6/25 12:47
# @Author: william.cao
# @File  : main.py.py
import logging
from concurrent.futures import ThreadPoolExecutor

import asyncio

from dingding import setup_redis
from log_processor import consume_kafka_messages
from model_training import train_model
from es_to_kafka import es_send_to_kafka

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_es_send_to_kafka_in_thread():
    executor = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, es_send_to_kafka)

async def train_model_clear(csv_file, num_epochs):
    # 模型训练代码
    logging.info(f"开始训练模型，使用文件 {csv_file}，训练次数 {num_epochs}")
    await train_model(csv_file=csv_file, num_epochs=num_epochs)
    logging.info("模型训练完成")

    # 清空 CSV 文件
    with open(csv_file, 'w') as file:
        file.truncate()
    logging.info(f"文件 {csv_file} 已清空")

async def main():
    try:
        await setup_redis()  # 初始化 Redis 客户端
        loop = asyncio.get_event_loop()
        
        polling_interval = 300  # 5分钟(300秒)
        
        while True:
            # 创建并运行Kafka消费和ES发送任务
            await loop.run_in_executor(None, consume_kafka_messages)  # 直接等待Kafka消费任务完成
            await loop.run_in_executor(None, es_send_to_kafka)
            
            # 等待5分钟
            await asyncio.sleep(polling_interval)
                
    except Exception as e:
        logging.error(f"程序运行失败: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("程序被中断")
    except Exception as e:
        logging.error(f"程序运行失败: {e}")
    finally:
        logging.info("程序结束")

# import logging
# from concurrent.futures import ThreadPoolExecutor
# import asyncio
# from model_training import train_model
# from es_to_kafka import es_send_to_kafka
#
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#
# async def train_model_clear(csv_file, num_epochs):
#     # 模型训练代码
#     logging.info(f"开始训练模型，使用文件 {csv_file}，训练次数 {num_epochs}")
#     await train_model(csv_file=csv_file, num_epochs=num_epochs)
#     logging.info("模型训练完成")
#
#     # 清空 CSV 文件
#     with open(csv_file, 'w') as file:
#         file.truncate()
#     logging.info(f"文件 {csv_file} 已清空")
