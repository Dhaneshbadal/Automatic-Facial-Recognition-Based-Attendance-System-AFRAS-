# recognition/web_interface.py
"""
Web interface integration for Django
Implements: real-time feed, reports, and management dashboard
"""
import base64
import json
import cv2
import numpy as np
from datetime import datetime
from .recognizer import identify_with_frame, get_registered_students
from .attendance_logger import AttendanceLogger

class WebAttendanceHandler:
    """
    Handles web requests for attendance system
    """
    def __init__(self):
        self.logger = AttendanceLogger()
        self.current_session = None
    
    def start_session(self, course_name, subject):
        """Start a new attendance session"""
        self.current_session = self.logger.create_session(course_name, subject)
        return self.current_session
    
    def end_current_session(self):
        """End the current attendance session"""
        if self.current_session:
            self.logger.end_session(self.current_session)
            session = self.current_session
            self.current_session = None
            return session
        return None
    
    def process_frame(self, frame_data, threshold=0.5):
        """
        Process a frame from web interface
        frame_data: base64 encoded image
        Returns: list of recognized faces
        """
        # Decode base64 image
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]
        
        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Recognize faces
        results = identify_with_frame(frame, threshold=threshold)
        
        # Log attendance if session is active
        if self.current_session:
            for result in results:
                if result["is_known"] and result["confidence"] > 0.6:
                    self.logger.log_attendance(
                        self.current_session,
                        result["student_id"] if "student_id" in result else result["name"],
                        result["name"],
                        result["confidence"]
                    )
        
        return results
    
    def get_report(self, start_date=None, end_date=None):
        """Get attendance report"""
        return self.logger.get_attendance_report(start_date, end_date)
    
    def get_dashboard_data(self):
        """Get data for admin dashboard"""
        students = get_registered_students()
        today_attendance = self.logger.get_today_attendance()
        daily_summary = self.logger.get_daily_summary()
        
        return {
            'total_students': len(students),
            'today_present': len(today_attendance),
            'today_percentage': (len(today_attendance) / len(students) * 100) if students else 0,
            'high_confidence': daily_summary[1] if daily_summary else 0,
            'low_confidence': daily_summary[2] if daily_summary else 0,
            'students': students[:10]  # Top 10 students
        }