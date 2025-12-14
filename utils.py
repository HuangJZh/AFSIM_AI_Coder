import os
import logging
import yaml
from typing import Any, Optional
import threading

class ConfigManager:
    """配置管理器"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ConfigManager, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        """初始化配置管理器"""
        self.config = {}
        self._load_config()
        self._setup_logging()
    
    def _load_config(self):
        """加载配置文件"""
        config_paths = [
            "config.yaml",
            os.path.join(os.path.dirname(__file__), "config.yaml"),
            os.path.expanduser("~/.afsim_coder/config.yaml"),
            "/etc/afsim_coder/config.yaml"
        ]
        
        loaded = False
        for path in config_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        self.config = yaml.safe_load(f)
                    logging.info(f"加载配置文件: {path}")
                    loaded = True
                    break
                except Exception as e:
                    logging.warning(f"配置文件 {path} 加载失败: {e}")
        
        # 默认配置
        if not loaded:
            self.config = {
                'model': {
                    'path': "D:/Qwen/Qwen/Qwen3-4B",
                    'max_tokens': 4096,
                    'temperature': 0.2,
                    'generation': {
                        'max_new_tokens': 1024,
                        'temperature': 0.3,
                        'top_p': 0.9,
                        'do_sample': True,
                        'repetition_penalty': 1.1
                    }
                },
                'vector_db': {
                    'chunk_size': 1500,
                    'chunk_overlap': 250,
                    'persist_dir': "vector_db_afsim_enhanced",
                    'collection_name': "afsim_tutorials"
                },
                'embedding': {
                    'model_name': "BAAI/bge-small-zh-v1.5",
                    'normalize_embeddings': True,
                    'batch_size': 32
                },
                'database': {
                    'chroma_path': "./chroma_db",
                    'settings': {
                        'anonymized_telemetry': False
                    }
                },
                'document': {
                    'supported_extensions': [".md", ".txt"],
                    'max_file_size_mb': 10,
                    'default_chunk_size': 400
                },
                'web': {
                    'port': 7860,
                    'share': False,
                    'debug': True
                },
                'logging': {
                    'level': "INFO",
                    'format': "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    'file': "afsim_rag.log"
                },
                'paths': {
                    'tutorials_folder': "tutorials",
                    'documents_list': "tree_of_tutorials.txt"
                },
                'system': {
                    'device_map': "auto",
                    'torch_dtype': "float16",
                    'load_in_4bit': True,
                    'use_quantization': True
                }
            }
            logging.warning("使用默认配置")
    
    def _setup_logging(self):
        """设置日志"""
        log_config = self.get('logging', {})
        log_level = log_config.get('level', 'INFO')
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = log_config.get('file')
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                *([logging.FileHandler(log_file)] if log_file else [])
            ]
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k)
                else:
                    return default
            
            return value if value is not None else default
        except (AttributeError, TypeError):
            return default
    
    def get_int(self, key: str, default: int = 0) -> int:
        """获取整型配置值"""
        value = self.get(key, default)
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """获取浮点型配置值"""
        value = self.get(key, default)
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """获取布尔型配置值"""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        elif isinstance(value, str):
            return value.lower() in ['true', 'yes', '1', 'on']
        elif isinstance(value, (int, float)):
            return bool(value)
        return default