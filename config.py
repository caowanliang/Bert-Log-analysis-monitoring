#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2024/6/21 11:24
# @Author: william.cao
# @File  : config.py.py
import json


ES_QUARRY_DATA = {"_source":["@timestamp","kubernetes","message"],"size":8000,"query":{"bool":{"should":[{"match_phrase":{"message":"Exception"}},{"match_phrase":{"message":"Failed"}}],"minimum_should_match":1,"filter":[{"term":{"kubernetes.namespace":"pbc"}}],"must":[{"range":{"@timestamp":{"gte":"now-5m/m"}}}]}}}

class Config:
    KAFKA_BROKERS = []
    KAFKA_TOPIC = ''
    NAMESPACES = []
    CONTAINER_NAME_PREFIXES = []
    DINGTALK_WEBHOOK = ''
    SECRET = ''
    REDIS_HOST = ''
    REDIS_PORT = 0
    REDIS_PWD = ''
    ES_URL = ''
    ES_AUTH = ('', '')
    ES_QUERY = '{"_source":["@timestamp","kubernetes","message"],"size":3000,"query":{"bool":{"should":[{"match_phrase":{"message":"Exception"}},{"match_phrase":{"message":"Failed"}}],"minimum_should_match":1,"filter":[{"term":{"kubernetes.namespace":"pbc"}}],"must":[{"range":{"@timestamp":{"gte":"now-5m/m"}}}]}}}'
