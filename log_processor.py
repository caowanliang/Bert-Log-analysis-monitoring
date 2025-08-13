#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2024/6/25 12:46
# @Author: william.cao
# @File  : log_processor.py
# 日志处理器

import logging
import json
from asyncio import as_completed

from kafka import KafkaConsumer
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_fixed
import asyncio
from config import Config
from log_processing import process_log

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_log_message(message):
    """
    解析 Kafka 消息，将消息转换为 JSON 对象。
    :param message: 原始消息，可能是字节或字符串
    :return: 解析后的 JSON 对象或 None
    """
    try:
        if not message:
            logging.warning("收到空消息")
            return None

        if isinstance(message, bytes):
            message = message.decode('utf-8')

        log_entry = json.loads(message)
        return log_entry
    except json.JSONDecodeError as e:
        logging.error(f"JSON 解码失败: {e}")
        return None

def process_logs_async(logs):
    """
    多线程异步处理日志列表。
    :param logs: 日志列表
    """
    # 创建一个新的事件循环
    loop = asyncio.new_event_loop()
    # 将新创建的事件循环设置为当前事件循环
    asyncio.set_event_loop(loop)
    try:
        # 创建任务列表，处理每个日志
        tasks = [process_log(log) for log in logs]
        # 运行所有任务直到完成
        loop.run_until_complete(asyncio.gather(*tasks))
    finally:
        # 关闭事件循环
        loop.close()

@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
def consume_kafka_messages():
    """
    消费 Kafka 消息并将日志信息提交给处理函数。
    """
    try:
        logging.info("开始连接 Kafka 服务器...")
        consumer = KafkaConsumer(
            Config.KAFKA_TOPIC,  # 替换为实际的 Kafka 主题
            bootstrap_servers=Config.KAFKA_BROKERS,  # 替换为实际的 Kafka brokers
            auto_offset_reset='latest',
            enable_auto_commit=True,  # 启用自动提交偏移量
            group_id='mss_log_cwl',
            value_deserializer=lambda x: parse_log_message(x))
        logging.info("Kafka 服务器连接成功，开始消费消息...")

        with ThreadPoolExecutor() as executor:
            future_to_log = {}
            for message in consumer:
                if message.value:
                    # logging.info(f"收到消息: {message.value}")
                    future = executor.submit(process_logs_async, [message.value])
                    future_to_log[future] = message.value
                else:
                    logging.warning(f"空消息或无法解析: {message.value}")

            # 处理已完成的任务
            for future in as_completed(future_to_log):
                log_value = future_to_log[future]
                try:
                    future.result()
                    logging.info(f"任务完成: {log_value}")
                except Exception as e:
                    logging.error(f"任务失败: {log_value}, 错误: {e}")
    except Exception as e:
        logging.error(f"消费 Kafka 消息时遇到错误: {e}")
