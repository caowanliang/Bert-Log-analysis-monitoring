#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2024/7/3 18:39
# @Author: william.cao
# @File  : client_redis.py
import aioredis
import logging

import asyncio

stop_event = asyncio.Event()
tasks = []
redis_client = None

async def initialize_redis_client(host, port, db, password):
    """
    初始化并返回Redis客户端
    :param host: Redis服务器主机名
    :param port: Redis服务器端口
    :param db: Redis数据库索引
    :param password: Redis服务器密码
    :return: Redis客户端实例
    """
    try:
        client = await aioredis.from_url(
            f"redis://{host}:{port}/{db}",
            password=password,
            encoding="utf-8",
            decode_responses=True
        )
        # 测试连接
        await client.ping()
        logging.info("Connected to Redis successfully")
        return client
    except aioredis.RedisError as e:
        logging.error(f"Failed to connect to Redis: {e}")
        raise

async def close_redis_client(client):
    """
    关闭Redis客户端连接
    :param client: Redis客户端实例
    """
    try:
        await client.close()
        logging.info("Redis连接已关闭")
    except aioredis.RedisError as e:
        logging.error(f"关闭Redis连接失败: {e}")
        raise
