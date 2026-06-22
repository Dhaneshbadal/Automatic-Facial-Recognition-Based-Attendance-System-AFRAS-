"""
Custom Face Recognition Based Attendance System
Features:
- High accuracy custom face recognition model
- Continuous tracking of students during class
- Attendance marked only when present for 85% of session time
- Threshold-based matching (0.6)
- Image processing pipeline
"""

import cv2
import numpy as np
import dlib
import pickle
import sqlite3
from datetime import datetime, timedelta
from collections import defaultdict
import os
from pathlib import Path

class FaceRecognitionAttendanceSystem:
    def __init__(self, 
                 face_recognition_threshold=0.6,
                 attendance_threshold=0.85,
                 tracking_interval=1):
        """
        Initialize the attendance system
        
        Args:
            face_recognition_threshold: Maximum distance for face matching (default: 0.6)
            attendance_threshold: Minimum presence ratio for attendance (default: 0.85 = 85%)
            tracking_interval: Seconds between tracking updates (default: 1)
        """
        self.face_threshold = face_recognition_threshold
        self.attendance_threshold = attendance_threshold
        self.tracking_interval = tracking_interval
        
        # Initialize face detection and recognition models
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
        
        # Also use Haar Cascade for additional detection
        self.haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Student tracking data
        self.registered_faces = {}  # {student_id: face_encodings_list}
        self.session_tracking = defaultdict(list)  # {student_id: [timestamps]}
        self.session_start_time = None
        self.session_duration = None
        
        # Database setup
        self.db_path = "attendance_system.db"
        self.setup_database()
        
    def setup_database(self):
        """Setup SQLite database for attendance records"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Students table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                enrollment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Face encodings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_encodings (
                encoding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                encoding BLOB,
                FOREIGN KEY (student_id) REFERENCES students(student_id)
            )
        ''')
        
        # Attendance records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                session_date DATE,
                session_start_time TIMESTAMP,
                session_end_time TIMESTAMP,
                presence_duration INTEGER,
                presence_percentage REAL,
                status TEXT,
                FOREIGN KEY (student_id) REFERENCES students(student_id)
            )
        ''')
        
        # Session tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_tracking (
                tracking_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                session_date DATE,
                timestamp TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students(student_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def preprocess_image(self, image):
        """
        Advanced image preprocessing for better face recognition
        
        Args:
            image: Input BGR image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale for some operations
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Histogram equalization for better contrast
        gray = cv2.equalizeHist(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Convert back to BGR for dlib
        preprocessed = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        
        # Gamma correction for lighting normalization
        gamma = 1.2
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        preprocessed = cv2.LUT(preprocessed, table)
        
        return preprocessed
    
    def detect_faces(self, image):
        """
        Detect faces using both Dlib and Haar Cascade for robustness
        
        Args:
            image: Input image
            
        Returns:
            List of face rectangles
        """
        # Preprocess image
        preprocessed = self.preprocess_image(image)
        
        # Method 1: Dlib HOG detector
        dlib_faces = self.detector(preprocessed, 1)
        
        # Method 2: Haar Cascade
        gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
        haar_faces = self.haar_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        # Combine detections (prefer dlib as it's more accurate)
        faces = []
        
        # Add dlib detections
        for face in dlib_faces:
            faces.append(face)
        
        # Add haar detections not overlapping with dlib
        for (x, y, w, h) in haar_faces:
            haar_rect = dlib.rectangle(x, y, x+w, y+h)
            overlap = False
            for dlib_face in dlib_faces:
                if self._rectangles_overlap(haar_rect, dlib_face):
                    overlap = True
                    break
            if not overlap:
                faces.append(haar_rect)
        
        return faces
    
    def _rectangles_overlap(self, rect1, rect2):
        """Check if two rectangles overlap"""
        return not (rect1.right() < rect2.left() or 
                   rect1.left() > rect2.right() or 
                   rect1.bottom() < rect2.top() or 
                   rect1.top() > rect2.bottom())
    
    def get_face_encoding(self, image, face_rect):
        """
        Generate 128-dimensional face encoding
        
        Args:
            image: Input image
            face_rect: Face rectangle from detector
            
        Returns:
            128-dimensional numpy array
        """
        # Get facial landmarks
        shape = self.shape_predictor(image, face_rect)
        
        # Generate face encoding
        face_encoding = self.face_encoder.compute_face_descriptor(image, shape, num_jitters=10)
        
        return np.array(face_encoding)
    
    def register_student(self, student_id, name, image_paths):
        """
        Register a new student with multiple face images
        
        Args:
            student_id: Unique student identifier
            name: Student name
            image_paths: List of image file paths for the student
        """
        encodings = []
        
        for img_path in image_paths:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not load image {img_path}")
                continue
            
            # Preprocess
            preprocessed = self.preprocess_image(image)
            
            # Detect faces
            faces = self.detect_faces(preprocessed)
            
            if len(faces) == 0:
                print(f"Warning: No face detected in {img_path}")
                continue
            
            # Use the largest face
            face = max(faces, key=lambda r: r.width() * r.height())
            
            # Get encoding
            encoding = self.get_face_encoding(preprocessed, face)
            encodings.append(encoding)
        
        if len(encodings) < 3:
            print(f"Warning: Only {len(encodings)} valid face images for {student_id}. Recommended: 5+")
        
        # Store in memory
        self.registered_faces[student_id] = encodings
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert student
        cursor.execute('''
            INSERT OR REPLACE INTO students (student_id, name)
            VALUES (?, ?)
        ''', (student_id, name))
        
        # Insert encodings
        for encoding in encodings:
            cursor.execute('''
                INSERT INTO face_encodings (student_id, encoding)
                VALUES (?, ?)
            ''', (student_id, pickle.dumps(encoding)))
        
        conn.commit()
        conn.close()
        
        print(f"Successfully registered {name} ({student_id}) with {len(encodings)} face encodings")
    
    def load_registered_faces(self):
        """Load all registered face encodings from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT student_id, encoding FROM face_encodings
        ''')
        
        self.registered_faces.clear()
        for student_id, encoding_blob in cursor.fetchall():
            encoding = pickle.loads(encoding_blob)
            if student_id not in self.registered_faces:
                self.registered_faces[student_id] = []
            self.registered_faces[student_id].append(encoding)
        
        conn.close()
        print(f"Loaded {len(self.registered_faces)} registered students")
    
    def recognize_face(self, face_encoding):
        """
        Recognize a face by comparing with registered encodings
        
        Args:
            face_encoding: 128-dimensional face encoding
            
        Returns:
            Tuple of (student_id, distance) or (None, None) if no match
        """
        best_match_id = None
        best_distance = float('inf')
        
        for student_id, encodings in self.registered_faces.items():
            # Compare with all encodings of this student
            for registered_encoding in encodings:
                # Euclidean distance
                distance = np.linalg.norm(face_encoding - registered_encoding)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match_id = student_id
        
        # Check if best match is within threshold
        if best_distance <= self.face_threshold:
            return best_match_id, best_distance
        else:
            return None, None
    
    def start_session(self, duration_minutes):
        """
        Start a new attendance session
        
        Args:
            duration_minutes: Total duration of the class session
        """
        self.session_start_time = datetime.now()
        self.session_duration = timedelta(minutes=duration_minutes)
        self.session_tracking.clear()
        
        print(f"Session started at {self.session_start_time.strftime('%H:%M:%S')}")
        print(f"Duration: {duration_minutes} minutes")
        print(f"Attendance threshold: {self.attendance_threshold * 100}%")
    
    def track_student_presence(self, student_id):
        """
        Record student presence at current timestamp
        
        Args:
            student_id: Student identifier
        """
        current_time = datetime.now()
        
        # Check if session is active
        if self.session_start_time is None:
            print("Error: No active session")
            return
        
        # Record timestamp
        self.session_tracking[student_id].append(current_time)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO session_tracking (student_id, session_date, timestamp)
            VALUES (?, ?, ?)
        ''', (student_id, current_time.date(), current_time))
        
        conn.commit()
        conn.close()
    
    def process_frame(self, frame):
        """
        Process a single video frame for face recognition and tracking
        
        Args:
            frame: BGR image from webcam
            
        Returns:
            Annotated frame with recognized faces
        """
        # Preprocess
        preprocessed = self.preprocess_image(frame)
        
        # Detect faces
        faces = self.detect_faces(preprocessed)
        
        recognized_students = []
        
        for face_rect in faces:
            # Get encoding
            try:
                encoding = self.get_face_encoding(preprocessed, face_rect)
                
                # Recognize
                student_id, distance = self.recognize_face(encoding)
                
                if student_id:
                    # Track presence
                    self.track_student_presence(student_id)
                    recognized_students.append((student_id, distance))
                    
                    # Draw rectangle and label
                    x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
                    
                    # Green for recognized
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Label
                    label = f"{student_id} ({distance:.2f})"
                    cv2.putText(frame, label, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    # Red for unrecognized
                    x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
            except Exception as e:
                print(f"Error processing face: {e}")
        
        return frame, recognized_students
    
    def calculate_presence_percentage(self, student_id):
        """
        Calculate the percentage of time a student was present
        
        Args:
            student_id: Student identifier
            
        Returns:
            Presence percentage (0-1)
        """
        if student_id not in self.session_tracking:
            return 0.0
        
        timestamps = self.session_tracking[student_id]
        
        if len(timestamps) == 0:
            return 0.0
        
        # Count unique seconds present
        unique_seconds = len(set(t.replace(microsecond=0) for t in timestamps))
        
        # Total session seconds
        total_seconds = self.session_duration.total_seconds()
        
        # Calculate percentage
        presence_percentage = unique_seconds / total_seconds
        
        return min(presence_percentage, 1.0)  # Cap at 100%
    
    def end_session(self):
        """
        End the current session and mark attendance
        
        Returns:
            Dictionary of attendance records
        """
        if self.session_start_time is None:
            print("Error: No active session")
            return {}
        
        session_end_time = datetime.now()
        attendance_records = {}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Process each tracked student
        for student_id in self.session_tracking.keys():
            presence_percentage = self.calculate_presence_percentage(student_id)
            presence_duration = int(presence_percentage * self.session_duration.total_seconds())
            
            # Determine status
            if presence_percentage >= self.attendance_threshold:
                status = "PRESENT"
            else:
                status = "ABSENT"
            
            attendance_records[student_id] = {
                'presence_percentage': presence_percentage * 100,
                'presence_duration': presence_duration,
                'status': status
            }
            
            # Insert into database
            cursor.execute('''
                INSERT INTO attendance (
                    student_id, session_date, session_start_time, session_end_time,
                    presence_duration, presence_percentage, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                student_id,
                self.session_start_time.date(),
                self.session_start_time,
                session_end_time,
                presence_duration,
                presence_percentage * 100,
                status
            ))
        
        conn.commit()
        conn.close()
        
        # Print summary
        print("\n" + "="*60)
        print("SESSION ENDED - ATTENDANCE SUMMARY")
        print("="*60)
        print(f"Session Duration: {self.session_duration.total_seconds()/60:.0f} minutes")
        print(f"Attendance Threshold: {self.attendance_threshold * 100}%")
        print("-"*60)
        
        for student_id, record in attendance_records.items():
            print(f"{student_id}: {record['status']} "
                  f"({record['presence_percentage']:.1f}% - "
                  f"{record['presence_duration']//60}m {record['presence_duration']%60}s)")
        
        print("="*60 + "\n")
        
        # Reset session
        self.session_start_time = None
        self.session_duration = None
        
        return attendance_records
    
    def run_live_attendance(self, duration_minutes=60, camera_index=0):
        """
        Run live attendance tracking
        
        Args:
            duration_minutes: Class duration in minutes
            camera_index: Camera device index
        """
        # Load registered faces
        self.load_registered_faces()
        
        # Start session
        self.start_session(duration_minutes)
        
        # Open camera
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Press 'q' to end session early")
        
        last_process_time = datetime.now()
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Check if session time is over
            elapsed = datetime.now() - self.session_start_time
            if elapsed >= self.session_duration:
                print("\nSession time completed!")
                break
            
            # Process frame every tracking_interval seconds
            current_time = datetime.now()
            if (current_time - last_process_time).total_seconds() >= self.tracking_interval:
                annotated_frame, recognized = self.process_frame(frame.copy())
                last_process_time = current_time
                
                # Display time remaining
                time_remaining = self.session_duration - elapsed
                minutes_remaining = int(time_remaining.total_seconds() // 60)
                seconds_remaining = int(time_remaining.total_seconds() % 60)
                
                cv2.putText(annotated_frame, 
                           f"Time Remaining: {minutes_remaining:02d}:{seconds_remaining:02d}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Attendance System', annotated_frame)
            else:
                # Just show the frame
                cv2.imshow('Attendance System', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nSession ended manually")
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # End session and show attendance
        self.end_session()


# Example usage
if __name__ == "__main__":
    # Initialize system
    system = FaceRecognitionAttendanceSystem(
        face_recognition_threshold=0.6,
        attendance_threshold=0.85,
        tracking_interval=1
    )
    
    # Register students (example)
    # system.register_student(
    #     student_id="BCE001",
    #     name="Dammara Thagunna",
    #     image_paths=["student_images/dammara_1.jpg", 
    #                  "student_images/dammara_2.jpg",
    #                  "student_images/dammara_3.jpg"]
    # )
    
    # Run live attendance for 60-minute class
    # system.run_live_attendance(duration_minutes=60, camera_index=0)
