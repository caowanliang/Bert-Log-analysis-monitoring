#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2024/7/22 15:51
# @Author: william.cao
# @File  : save_log_cache.py
import time


class SimpleCache:
    def __init__(self):
        self.cache = {}
        self.expiry = {}

    def set(self, key, value, ttl):
        self.cache[key] = value
        self.expiry[key] = time.time() + ttl

    def get(self, key):
        if key in self.cache and time.time() < self.expiry[key]:
            return self.cache[key]
        elif key in self.cache:
            del self.cache[key]
            del self.expiry[key]
        return None

    def clear_expired(self):
        current_time = time.time()
        keys_to_delete = [key for key, exp_time in self.expiry.items() if exp_time < current_time]
        for key in keys_to_delete:
            del self.cache[key]
            del self.expiry[key]
