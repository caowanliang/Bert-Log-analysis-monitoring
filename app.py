import json
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import logging
import threading
import asyncio
from dingding import setup_redis, close_redis
from es_to_kafka import es_send_to_kafka
from forms import ConfigForm
from config import Config
from log_processor import consume_kafka_messages
from main import train_model_clear
from model_training import load_config, save_config

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

main_thread = None
is_main_thread_running = False
stop_event = threading.Event()
tasks = []

# 加载配置
app_config = load_config()

@app.route('/', methods=['GET', 'POST'])
def index():
    form = ConfigForm()

    if request.method == 'POST':
        if form.validate_on_submit():
            Config.KAFKA_BROKERS = form.kafka_brokers.data.split(',')
            Config.KAFKA_TOPIC = form.kafka_topic.data
            Config.NAMESPACES = form.namespaces.data.split(',')
            Config.CONTAINER_NAME_PREFIXES = form.container_name_prefixes.data.split(',') if form.container_name_prefixes.data else []
            Config.DINGTALK_WEBHOOK = form.dingtalk_webhook.data
            Config.SECRET = form.secret.data
            Config.REDIS_HOST = form.redis_host.data
            Config.REDIS_PORT = int(form.redis_port.data)
            Config.REDIS_PWD = form.redis_pwd.data
            Config.ES_URL = form.es_url.data
            Config.ES_AUTH = (form.es_username.data, form.es_password.data)

            flash('Configuration updated!', 'success')
            return redirect(url_for('buttons'))

    return render_template('config.html', form=form)

@app.route('/buttons', methods=['GET', 'POST'])
def buttons():
    global is_main_thread_running

    if request.method == 'POST':
        ignore_keywords = request.form.get('ignore_keywords', '').split(',')
        increase_keywords = request.form.get('increase_keywords', '').split(',')
        decrease_keywords = request.form.get('decrease_keywords', '').split(',')

        app_config['ignore_keywords'] = ignore_keywords
        app_config['specific_keywords_increase'] = increase_keywords
        app_config['specific_keywords_decrease'] = decrease_keywords

        save_config(app_config)
        flash('Configuration updated!', 'success')
        return redirect(url_for('buttons'))

    is_running = is_main_thread_running
    es_query = Config.ES_QUERY
    return render_template('buttons.html', is_running=is_running, es_query=es_query, config=app_config)

@app.route('/start', methods=['POST'])
def start():
    global main_thread, is_main_thread_running
    try:
        es_query = request.form['es_query']
        json.loads(es_query)  # 尝试解析 JSON

        Config.ES_QUERY = es_query  # 更新配置

        if not is_main_thread_running:
            stop_event.clear()
            main_thread = threading.Thread(target=run_main)
            main_thread.start()
            is_main_thread_running = True
            flash('Main process started', 'success')
        else:
            flash('Main process is already running', 'warning')
    except json.JSONDecodeError:
        flash('查询语句不正确，请输入有效的 JSON 格式', 'danger')

    return redirect(url_for('buttons'))

@app.route('/train', methods=['POST'])
def train():
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(train_model_now())
        return jsonify(message="AI训练完成")
    except Exception as e:
        logging.error(f"AI训练失败: {e}")
        if "No columns to parse from file" in str(e):
            return jsonify(message="文件为空"), 400
        return jsonify(message="AI训练失败"), 500

async def train_model_now():
    await train_model_clear(csv_file='logs.csv', num_epochs=3)

def run_main():
    asyncio.run(main())

async def main():
    global tasks
    try:
        await setup_redis()
        kafka_task = asyncio.create_task(consume_kafka_messages_wrapper())
        es_task = asyncio.create_task(es_send_to_kafka_wrapper())
        tasks = [kafka_task, es_task]

        while not stop_event.is_set():
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        logging.info("Tasks have been cancelled.")
    except Exception as e:
        logging.error(f"程序运行失败: {e}")
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await close_redis()
        tasks = []
        logging.info("程序结束")

async def consume_kafka_messages_wrapper():
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, consume_kafka_messages)
    except Exception as e:
        logging.error(f"Kafka consumer error: {e}")

async def es_send_to_kafka_wrapper():
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, es_send_to_kafka)
    except Exception as e:
        logging.error(f"Elasticsearch to Kafka error: {e}")

@app.route('/save_config', methods=['POST'])
def save_config():
    form = ConfigForm()
    
    if form.validate_on_submit():
        # 更新配置
        Config.KAFKA_BROKERS = form.kafka_brokers.data.split(',')
        Config.KAFKA_TOPIC = form.kafka_topic.data
        Config.NAMESPACES = form.namespaces.data.split(',')
        Config.CONTAINER_NAME_PREFIXES = form.container_name_prefixes.data.split(',') if form.container_name_prefixes.data else []
        Config.DINGTALK_WEBHOOK = form.dingtalk_webhook.data
        Config.SECRET = form.secret.data
        Config.REDIS_HOST = form.redis_host.data
        Config.REDIS_PORT = int(form.redis_port.data)
        Config.REDIS_PWD = form.redis_pwd.data
        Config.ES_URL = form.es_url.data
        Config.ES_AUTH = (form.es_username.data, form.es_password.data)
        
        flash('配置保存成功', 'success')
    else:
        flash('配置保存失败，请检查输入', 'danger')
    
    return redirect(url_for('buttons'))

@app.route('/test_redis', methods=['POST'])
def test_redis():
    try:
        import redis
        data = request.get_json()
        r = redis.Redis(
            host=data.get('host', 'localhost'),
            port=int(data.get('port', 6379)),
            password=data.get('password', None),
            socket_timeout=3  # 设置3秒超时
        )
        r.ping()  # 测试连接
        return jsonify({'success': True, 'message': 'Redis连接正常'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
