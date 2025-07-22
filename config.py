import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key')
    MODEL_CACHE_DIR = os.environ.get('MODEL_CACHE_DIR', './model_cache')