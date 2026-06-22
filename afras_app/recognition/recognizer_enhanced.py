# recognition/recognizer_enhanced.py
"""
Enhanced recognizer integrating all modules
"""
import numpy as np
import cv2
import face_recognition
from .config import Config
from .attendance_logger import AttendanceLogger
from .liveness import LivenessDetector
from .performance_optimizer import AdaptiveFrameProcessor
from .face_aligner import FaceAligner

class EnhancedFaceRecognizer:
    """
    Enhanced face recognizer with all features from proposal
    """
    def __init__(self):
        self.config = Config()
        self.logger = AttendanceLogger(self.config.get("db_path"))
        self.liveness = LivenessDetector()
        self.performance = AdaptiveFrameProcessor(
            target_fps=self.config.get("target_fps"),
            resize_factor=self.config.get("resize_factor")
        )
        self.aligner = FaceAligner()
        
        self.load_model()
    
    def load_model(self):
        """Load trained model"""
        import pickle
        from pathlib import Path
        
        model_path = Path(self.config.get("model_path"))
        
        if model_path.exists():
            with open(model_path, "rb") as f:
                data = pickle.loads(f.read())
                self.known_encodings = np.array(data.get("encodings", []))
                self.known_names = np.array(data.get("names", []))
                self.known_ids = np.array(data.get("ids", []))
        else:
            self.known_encodings = np.array([])
            self.known_names = np.array([])
            self.known_ids = np.array([])
    
    def process_frame(self, frame, session_id=None, threshold=None):
        """
        Process a single frame with all enhancements
        """
        if threshold is None:
            threshold = self.config.get("recognition_threshold")
        
        # Adaptive resize based on performance
        resize_factor = self.performance.update(0)  # Will be updated with actual time
        
        # Detect faces
        small_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(
            rgb_small, 
            model=self.config.get("detection_model")
        )
        
        face_encodings = face_recognition.face_encodings(
            rgb_small, 
            face_locations,
            num_jitters=self.config.get("num_jitters"),
            model=self.config.get("encoding_model")
        )
        
        # Scale coordinates back
        scale = 1.0 / resize_factor
        results = []
        
        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            # Scale coordinates
            top, right, bottom, left = [
                int(coord * scale) for coord in [top, right, bottom, left]
            ]
            
            # Recognize
            name, student_id, confidence = self._recognize_face(encoding, threshold)
            
            # Liveness check if enabled
            is_live = True
            live_score = 1.0
            
            if self.config.get("liveness_check"):
                is_live, live_score = self.liveness.is_live(frame, (top, right, bottom, left))
            
            # Only log if live and confidence is high enough
            if session_id and is_live and confidence >= self.config.get("min_confidence"):
                self.logger.log_attendance(
                    session_id, 
                    student_id or name, 
                    name, 
                    confidence,
                    self.config.get("attendance_cooldown")
                )
            
            results.append({
                "name": name,
                "student_id": student_id,
                "confidence": confidence,
                "is_live": is_live,
                "live_score": live_score,
                "coords": (top, right, bottom, left)
            })
        
        return results
    
    def _recognize_face(self, face_encoding, threshold):
        """Recognize a single face encoding"""
        name = "Unknown"
        student_id = None
        confidence = 0.0
        
        if len(self.known_encodings) > 0:
            distances = np.linalg.norm(self.known_encodings - face_encoding, axis=1)
            best_idx = np.argmin(distances)
            best_dist = distances[best_idx]
            
            confidence = max(0, min(1, 1 - (best_dist / 0.8)))
            
            if best_dist <= threshold:
                name = self.known_names[best_idx]
                if len(self.known_ids) > best_idx:
                    student_id = self.known_ids[best_idx]
        
        return name, student_id, confidence
    
    def start_attendance_session(self, course_name="", subject=""):
        """Start a new attendance session"""
        session_id = self.logger.create_session(course_name, subject)
        return session_id
    
    def end_attendance_session(self, session_id):
        """End an attendance session"""
        self.logger.end_session(session_id)
    
    def get_report(self, start_date=None, end_date=None):
        """Get attendance report"""
        return self.logger.get_attendance_report(start_date, end_date)
    
    def get_dashboard_stats(self):
        """Get dashboard statistics"""
        students = set(self.known_names)
        today_attendance = self.logger.get_today_attendance()
        
        return {
            "total_students": len(students),
            "today_present": len(today_attendance),
            "attendance_rate": (len(today_attendance) / len(students) * 100) if students else 0,
            "model_encodings": len(self.known_encodings)
        }