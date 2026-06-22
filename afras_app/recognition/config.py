# recognition/config.py
"""
Configuration management for AFRAS
"""
import json
import os
from pathlib import Path

class Config:
    def __init__(self, config_path="recognition/config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_defaults()
        
        if self.config_path.exists():
            self._load()
        else:
            self._save()
    
    def _load_defaults(self):
        """Default configuration"""
        return {
            # Recognition settings
            "recognition_threshold": 0.5,
            "detection_model": "hog",  # hog or cnn
            "encoding_model": "large",  # small or large
            "resize_factor": 0.25,
            "num_jitters": 1,
            
            # Performance settings
            "target_fps": 15,
            "frame_skip": 1,
            "use_gpu": False,
            
            # Security settings
            "liveness_check": True,
            "min_confidence": 0.6,
            "attendance_cooldown": 5,  # minutes
            
            # Database settings
            "db_path": "recognition/attendance.db",
            "model_path": "recognition/models/trained_faces.pkl",
            "dataset_path": "recognition/dataset/",
            
            # Training settings
            "use_augmentation": True,
            "min_encodings_per_student": 3,
            "augmentation_count": 4,
            
            # Web interface
            "web_host": "127.0.0.1",
            "web_port": 8000,
            "debug_mode": True,
            
            # Notification settings
            "email_notifications": False,
            "smtp_server": "",
            "smtp_port": 587,
            "admin_email": "",
            
            # Logging
            "log_level": "INFO",
            "log_file": "recognition/afras.log"
        }
    
    def _load(self):
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                loaded = json.load(f)
                self.config.update(loaded)
        except Exception as e:
            print(f"Error loading config: {e}")
    
    def _save(self):
        """Save configuration to file"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get(self, key, default=None):
        return self.config.get(key, default)
    
    def set(self, key, value):
        self.config[key] = value
        self._save()
    
    def update(self, updates):
        self.config.update(updates)
        self._save()