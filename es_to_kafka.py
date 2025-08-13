import json
import logging
import requests
import schedule
import time
from config import Config

from kafka import KafkaProducer

from config import ES_QUARRY_DATA


class KafkaProducerWrapper:
    def __init__(self, brokers, topic):
        self.producer = KafkaProducer(bootstrap_servers=brokers, value_serializer=lambda v: json.dumps(v).encode('utf-8'))
        self.topic = topic

    def send(self, message):
        # logging.info(f"发送消息到 Kafka: {message}")
        future = self.producer.send(self.topic, message)
        result = future.get(timeout=10)
        # logging.info(f"消息发送结果: {result}")

    def close(self):
        self.producer.close()



# 配置日志记录
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def query_es(kafka_producer):
    data = ES_QUARRY_DATA
    headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json'
}
    response = requests.post(Config.ES_URL, auth=Config.ES_AUTH, headers=headers, json=data)
    if response.status_code == 200:
        res = response.json()
        hits = res['hits']['hits']
        logging.info(f"Elasticsearch 查询成功，开始发送 {len(hits)} 条数据到 Kafka")
        for hit in hits:
            kafka_producer.send(hit['_source'])
        kafka_producer.producer.flush()  # 刷新缓冲区，确保消息发送出去
    else:
        logging.error(f"查询 Elasticsearch 失败，状态码: {response.status_code}, 响应内容: {response.content}")


def es_send_to_kafka():
    # 初始化 Kafka 生产者
    kafka_producer = KafkaProducerWrapper(Config.KAFKA_BROKERS, Config.KAFKA_TOPIC)
    # 定义调度任务
    schedule.every(5).minutes.do(query_es, kafka_producer)

    # 启动任务调度器
    try:
        logging.info("启动任务调度器")
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("停止任务调度器")
    finally:
        kafka_producer.close()
