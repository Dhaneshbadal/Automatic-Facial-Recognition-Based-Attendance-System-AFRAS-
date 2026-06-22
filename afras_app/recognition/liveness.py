"""
Liveness detection to prevent photo/video spoofing
Fixed version - no size mismatch errors
"""
import cv2
import numpy as np

class LivenessDetector:
    def __init__(self):
        # Eye blink detection parameters
        self.blink_counter = 0
        self.eye_aspect_ratio_history = []
        self.blink_threshold = 0.25
        
        # Motion detection parameters
        self.prev_frame = None
        self.motion_history = []
        
        # Load Haar cascades
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
    def calculate_eye_aspect_ratio(self, eye_points):
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection
        """
        if len(eye_points) < 6:
            return 0
        
        # Convert to numpy array
        eye_points = np.array(eye_points)
        
        # Vertical distances
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Horizontal distance
        h = np.linalg.norm(eye_points[0] - eye_points[3])
        
        if h == 0:
            return 0
        
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def detect_blink(self, face_landmarks):
        """Detect eye blinks using facial landmarks"""
        try:
            left_eye = face_landmarks.get('left_eye', [])
            right_eye = face_landmarks.get('right_eye', [])
            
            if left_eye and right_eye:
                left_ear = self.calculate_eye_aspect_ratio(left_eye)
                right_ear = self.calculate_eye_aspect_ratio(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                
                self.eye_aspect_ratio_history.append(avg_ear)
                
                # Keep only last 30 frames
                if len(self.eye_aspect_ratio_history) > 30:
                    self.eye_aspect_ratio_history.pop(0)
                
                # Detect blink (sudden drop in EAR)
                if len(self.eye_aspect_ratio_history) > 5:
                    if avg_ear < self.blink_threshold and \
                       max(self.eye_aspect_ratio_history[-5:]) > 0.3:
                        self.blink_counter += 1
                        return True
        except Exception as e:
            pass
        return False
    
    def detect_motion(self, face_roi):
        """
        Detect micro-motions to ensure live face
        Fixed: ensures same size frames
        """
        if face_roi is None or face_roi.size == 0:
            return 0
        
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize to fixed size for consistent comparison
        gray = cv2.resize(gray, (100, 100))
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return 0
        
        # Ensure both frames have same size
        if self.prev_frame.shape != gray.shape:
            self.prev_frame = cv2.resize(self.prev_frame, gray.shape[:2][::-1])
        
        # Calculate frame difference
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        motion_score = np.mean(frame_diff)
        
        self.prev_frame = gray
        self.motion_history.append(motion_score)
        
        if len(self.motion_history) > 10:
            self.motion_history.pop(0)
        
        return motion_score
    
    def detect_texture_analysis(self, face_roi):
        """
        Detect printed photo by analyzing texture
        """
        if face_roi.size == 0:
            return 0, 0
        
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate variance (texture)
        lbp_variance = np.var(gray)
        
        # Simple frequency analysis
        try:
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
            
            # Get center region for high frequencies
            h, w = magnitude_spectrum.shape
            center_h, center_w = h // 2, w // 2
            high_freq_energy = np.mean(magnitude_spectrum[center_h-30:center_h+30, 
                                                          center_w-30:center_w+30])
        except:
            high_freq_energy = 50
        
        return lbp_variance, high_freq_energy
    
    def is_live(self, frame, face_location, face_landmarks=None):
        """
        Determine if detected face is live (not a photo/video)
        Returns (is_live, confidence_score)
        """
        top, right, bottom, left = face_location
        
        # Ensure coordinates are valid
        h, w = frame.shape[:2]
        top = max(0, min(top, h-1))
        bottom = max(top+1, min(bottom, h))
        left = max(0, min(left, w-1))
        right = max(left+1, min(right, w))
        
        face_roi = frame[top:bottom, left:right]
        
        if face_roi.size == 0:
            return False, 0.0
        
        scores = []
        
        # 1. Eye detection (simple method - always works)
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 4)
        has_eyes = len(eyes) >= 1
        scores.append(0.4 if has_eyes else 0.1)
        
        # 2. Motion detection
        try:
            motion_score = self.detect_motion(face_roi)
            motion_confidence = min(0.3, motion_score / 30)
            scores.append(motion_confidence)
        except:
            scores.append(0.1)
        
        # 3. Texture analysis
        try:
            lbp_var, high_freq = self.detect_texture_analysis(face_roi)
            texture_confidence = 0.2 if lbp_var > 80 else 0.1
            scores.append(texture_confidence)
        except:
            scores.append(0.1)
        
        # 4. Check for blur
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_sharp = laplacian_var > 50
        scores.append(0.2 if is_sharp else 0.05)
        
        total_score = sum(scores)
        is_live = total_score > 0.5
        
        return is_live, total_score