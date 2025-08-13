#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2024/7/3 12:27
# @Author: william.cao
# @File  : dingding.py
import logging
import time
import requests
import json
import hmac
import hashlib
from urllib import parse
import base64

from client_redis import initialize_redis_client, close_redis_client
from config import Config
from error_analysis import preprocess_log_message
from log_processing import extract_error_log, redis_client_lock

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
redis_client = None

async def setup_redis():
    global redis_client
    redis_client = await initialize_redis_client(Config.REDIS_HOST, Config.REDIS_PORT, 14, Config.REDIS_PWD)

async def close_redis():
    global redis_client
    if redis_client:
        await close_redis_client(redis_client)

def create_sign(timestamp):
    """
    根据时间戳和密钥生成签名
    :param timestamp: 时间戳，单位为毫秒
    :return: 生成的签名字符串
    """
    string_to_sign = '{}\n{}'.format(timestamp, Config.SECRET)
    hmac_code = hmac.new(Config.SECRET.encode('utf-8'), string_to_sign.encode('utf-8'), hashlib.sha256).digest()
    sign = parse.quote_plus(base64.b64encode(hmac_code))
    return sign

async def send_dingtalk_notification(namespace_name, container_name, pod_name, message):
    """
    发送钉钉的消息
    :param pod_name: pod_name
    :param message: 日志消息
    :param namespace_name: 命名空间名称
    :param container_name: 容器名称
    """
    error_log, dbapp_log, error_time, thread_info= extract_error_log(message)

    # 移除error_log日志中的转义码
    error_log = preprocess_log_message(error_log)

    timestamp = str(round(time.time() * 1000))  # 当前时间的毫秒时间戳
    sign = create_sign(timestamp)

    headers = {
        'Content-Type': 'application/json; charset=utf-8',
        'timestamp': timestamp,
        'sign': sign
    }
    data = {
        "msgtype": "markdown",
        "markdown": {
            "title": "AI检测到异常日志",
            "text": f"### AI检测到异常日志，请关注！\n\n"
                    f"**错误时间:** {error_time}\n\n"
                    f"**空间名称:** {namespace_name}\n\n"
                    f"**容器名称:** {container_name}\n\n"
                    f"**节点名称:** {pod_name}\n\n"
                    f"**线程信息:** {thread_info}\n\n"
                    f"**错误日志:**\n```\n{error_log}\n```\n\n"
                    f"**相关日志:**\n```\n{dbapp_log}\n```\n"
                    f"[点击这里查询更多日志信息](http://10.50.26.46:5601/app/discover#/?_g=h@381543e&_a=h@6480a8d)"
        }
    }

    try:
        if error_log is not None and dbapp_log is not None:
            # 计算 log_message 的哈希值
            redis_error_log = preprocess_log_message(error_log)
            is_duplicate = await redis_client_lock(redis_error_log, 14, 43200, 720)

            if is_duplicate:
                logging.info("相同的错误日志已存在，不发送钉钉通知")
                return

            response = requests.post(Config.DINGTALK_WEBHOOK + '&timestamp=' + timestamp + '&sign=' + sign, headers=headers, data=json.dumps(data))
            if response.status_code == 200:
                logging.info("钉钉通知发送成功")
            else:
                logging.error(f"钉钉通知发送失败，状态码: {response.status_code}")

    except Exception as e:
        logging.error(f"发送钉钉通知时发生错误: {e}")
