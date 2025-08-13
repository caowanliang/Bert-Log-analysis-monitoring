#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2024/7/16 17:13
# @Author: william.cao
# @File  : forms.py

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, IntegerField
from wtforms.validators import DataRequired, Optional

class ConfigForm(FlaskForm):
    kafka_brokers = StringField('Kafka Brokers', validators=[DataRequired()], render_kw={"placeholder": "例如: 10.20.140.129:9091,10.20.140.130:9092"})
    kafka_topic = StringField('Kafka Topic', validators=[DataRequired()], render_kw={"placeholder": "例如: mss_ai_log"})
    namespaces = StringField('Namespaces', validators=[DataRequired()], render_kw={"placeholder": "例如: pbc,buss-platform"})
    container_name_prefixes = StringField('Container Name Prefixes', validators=[Optional()], render_kw={"placeholder": "例如: equipment-manage"})
    dingtalk_webhook = StringField('Dingtalk Webhook', validators=[DataRequired()], render_kw={"placeholder": "例如: https://oapi.dingtalk.com/robot/send?access_token=..."})
    secret = StringField('Secret', validators=[DataRequired()], render_kw={"placeholder": "例如: AI"})
    redis_host = StringField('Redis Host', validators=[DataRequired()], render_kw={"placeholder": "例如: 10.20.140.213"})
    redis_port = IntegerField('Redis Port', validators=[DataRequired()], render_kw={"placeholder": "例如: 16379"})
    redis_pwd = PasswordField('Redis Password', validators=[DataRequired()], render_kw={"placeholder": "输入Redis密码"})
    es_url = StringField('Elasticsearch URL', validators=[DataRequired()], render_kw={"placeholder": "例如: http://10.50.26.43:9200/kubernetes-prod-*/_search?scroll=1m"})
    es_username = StringField('Elasticsearch Username', validators=[DataRequired()], render_kw={"placeholder": "输入Elasticsearch用户名"})
    es_password = PasswordField('Elasticsearch Password', validators=[DataRequired()], render_kw={"placeholder": "输入Elasticsearch密码"})
    submit = SubmitField('提交')
