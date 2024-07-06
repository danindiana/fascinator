# main.py
from flask import Flask, jsonify
from app.crawlers.spider import run_spider
from app.ingest.tika_extractor import extract_text
from app.ingest.preprocessor import preprocess_text
from app.topic_model.model import train_topic_model, infer_topic
from app.clustering.clusterer import cluster_documents
from app.novelty.scorer import calculate_novelty_score
from app.cache.eviction import evict_cache
from app.agent.model import train_agent
from app.monitoring.metrics import track_metrics
from app.utils.db import connect_db
from app.utils.logging import setup_logging
from app.config import Config

app = Flask(__name__)

@app.route('/crawl', methods=['POST'])
def crawl():
    run_spider()
    return jsonify({"status": "crawling started"})

@app.route('/extract', methods=['POST'])
def extract():
    extract_text()
    return jsonify({"status": "text extraction started"})

@app.route('/preprocess', methods=['POST'])
def preprocess():
    preprocess_text()
    return jsonify({"status": "preprocessing started"})

@app.route('/train_topic_model', methods=['POST'])
def train_model():
    train_topic_model()
    return jsonify({"status": "topic model training started"})

@app.route('/infer_topic', methods=['POST'])
def infer():
    infer_topic()
    return jsonify({"status": "topic inference started"})

@app.route('/cluster', methods=['POST'])
def cluster():
    cluster_documents()
    return jsonify({"status": "clustering started"})

@app.route('/calculate_novelty', methods=['POST'])
def calculate_novelty():
    calculate_novelty_score()
    return jsonify({"status": "novelty score calculation started"})

@app.route('/evict_cache', methods=['POST'])
def evict():
    evict_cache()
    return jsonify({"status": "cache eviction started"})

@app.route('/train_agent', methods=['POST'])
def train_rl_agent():
    train_agent()
    return jsonify({"status": "RL agent training started"})

@app.route('/track_metrics', methods=['POST'])
def track():
    track_metrics()
    return jsonify({"status": "metrics tracking started"})

if __name__ == '__main__':
    setup_logging()
    connect_db()
    app.run(debug=Config.DEBUG)
