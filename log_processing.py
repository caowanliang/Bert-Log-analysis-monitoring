#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2024/7/22 11:14
# @Author: william.cao
# @File  : log_processing.py
# 日志预处理
import hashlib
import logging
import os
import re
from datetime import timedelta

from client_redis import close_redis_client, initialize_redis_client
from config import Config
from data_preparation import store_to_parquet

from error_analysis import preprocess_log_message, analyze_log_with_codebert
from save_log_cache import SimpleCache


async def process_log(log):
    """
    处理单条日志信息，筛选匹配的命名空间和容器名前缀，并提取错误日志。
    :param log: 日志对象
    """
    from dingding import send_dingtalk_notification
    if log:
        kubernetes_info = log.get("kubernetes", {})
        message = log.get("message", "")

        # 预处理日志消息，去除无关信息并提取有用特征
        log_message = preprocess_log_message(message)

        # 获取空间名称
        namespace_name = kubernetes_info.get("namespace", "")

        # 获取容器名称
        container_info = kubernetes_info.get("container", {})
        container_name = container_info.get("name", "")

        # 获取节点名称
        pod_info = kubernetes_info.get("pod", {})
        pod_name = pod_info.get("name", "")

        if namespace_name in Config.NAMESPACES:
            log_message_lower = log_message.lower()

            # 使用 BERT 模型分析日志消息
            confidence = await analyze_log_with_codebert(log_message_lower)

            # 忽略不需要关注的日志消息
            if confidence is None:
                return

            # 根据置信度给日志打标签
            logging.info(f"bert预测的置信度: {confidence}")
            label = 1 if confidence >= 0.7 else 0

            # 如果标签为 1，发送钉钉通知
            if label == 1:
                await send_dingtalk_notification(namespace_name, container_name, pod_name, message)

            # 处理实时日志，存储到文件
            await process_realtime_logs(log_message_lower, label)


# 实例化缓存对象
cache = SimpleCache()

async def process_realtime_logs(log_message, label):
    """
    实时处理日志，提取错误日志信息并标记后存储到 CSV 文件。
    :param confidence: 置信度
    :param log_message: 除去时间戳后的message
    :param label: 标记
    """
    try:
        error_log, dbapp_log, error_time, thread_info = extract_error_log(log_message)
        if error_log is not None and dbapp_log is not None:
            log_message = error_log + "-" + dbapp_log

        cache.clear_expired()  # 清理过期缓存
        is_duplicate = cache.get(log_message)

        if is_duplicate:
            logging.info("日志已存在，不重复存储")
            return

        # 设置缓存，TTL为300秒
        cache.set(log_message, True, 3000)

        # 存储日志到 Parquet 文件
        file_path = 'E:\\Bert-Log-analysis-monitoring\\Bert-Log-analysis-monitoring\\logs.parquet'
        logging.info(f"存储日志到 Parquet 文件地址: {file_path}")
        store_to_parquet(log_message, label, file_path)
        logging.info("日志已处理并存储")
    except Exception as e:
        logging.error(f"处理实时日志时发生错误: {e}")

def extract_error_log(log_message):
    """
    从日志消息中提取时间戳、错误日志信息和与 dbappsecurity 相关的信息。
    :param log_message: 日志信息字符串
    :return: 时间戳, 错误日志信息, dbappsecurity 相关日志
    """
    if not log_message or not isinstance(log_message, str):
        logging.error(f"日志消息不是字符串: {log_message}")
        return None, None, None

    log_lower = log_message.lower()  # 将日志消息转换为小写以忽略大小写匹配
    error_log = None
    dbapp_log = None
    time_stamp = None
    thread_info = None  # 用于存储线程信息

    # 定义常见错误关键词和正则表达式
    error_keywords = ["error", "warn", "failed", "exception", "sqlexception"]
    error_pattern = re.compile(r"(?P<type>exception|error|failed|warn).*?(?=\sat\s|$)", re.IGNORECASE | re.DOTALL)
    dbapp_pattern = re.compile(r'(com\.(dbappsecurity|mysql|alibaba)\.[^\s]+.*|###.*)', re.IGNORECASE)
    thread_pattern = re.compile(r'\[(?P<thread_info>[^\[\]]+)\]')  # 匹配方括号中的内容

    # 定义时间戳正则表达式模式
    timestamp_pattern = re.compile(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')

    # 使用正则表达式搜索日志消息中的时间戳
    match = timestamp_pattern.search(log_message)
    if match:
        time_stamp = match.group(0)

    # 使用正则表达式搜索日志消息中的线程信息
    thread_match = thread_pattern.findall(log_message)
    if thread_match:
        thread_info = [info for info in thread_match if 'worker' in info or 'exec' in info or 'C-' in info or 'JobThread' in info]
        if thread_info:
            thread_info = thread_info[0]
        else:
            thread_info = None

    # 检查日志中是否包含错误关键词
    if any(keyword in log_lower for keyword in error_keywords):
        match = error_pattern.search(log_message)
        if match:
            # 保留第一个 "at" 之前的所有内容
            error_log = match.group(0).strip()

    dbapp_match = dbapp_pattern.findall(log_message)
    if dbapp_match:
        dbapp_log = '\n'.join([log[0].strip() for log in dbapp_match])

    return error_log, dbapp_log, time_stamp, thread_info

async def redis_client_lock(log, redis_db, time_out, set_time):
    redis_client = await initialize_redis_client(Config.REDIS_HOST, Config.REDIS_PORT, redis_db, Config.REDIS_PWD)
    log_hash = hashlib.md5(log.encode('utf-8')).hexdigest()

    # 使用分布式锁进行去重检查
    lock_key = f"lock:{log_hash}"
    try:
        async with redis_client.lock(lock_key, timeout=time_out):
            # 检查键是否存在
            if await redis_client.exists(log_hash):
                logging.info(f"相同错误日志在指定时间内已处理过")
                return True  # 日志已存在

            # 设置键值并设置过期时间
            was_set = await redis_client.set(log_hash, "processed", ex=int(timedelta(minutes=set_time).total_seconds()))
            logging.info(f"redis去重的was_set: {was_set}")

            if not was_set:
                logging.error(f"Failed to set redis key: {log_hash}")
                return True  # 假设设置失败为已存在，避免重复处理
            return False
    finally:
        await close_redis_client(redis_client)
