"""
Image preprocessing for better face recognition accuracy
Handles lighting, noise, orientation, and quality enhancement
"""
import cv2
import numpy as np
from scipy import ndimage

class ImagePreprocessor:
    def __init__(self):
        self.face_detector = None  # Will initialize when needed
    
    def enhance_lighting(self, image):
        """
        Enhance lighting using histogram equalization and CLAHE
        Handles dark and overexposed images
        """
        if image is None:
            return None
        
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge back
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Apply gamma correction for brightness
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        enhanced = cv2.LUT(enhanced, table)
        
        return enhanced
    
    def remove_noise(self, image):
        """
        Remove noise while preserving edges
        """
        if image is None:
            return None
        
        # Apply bilateral filter (preserves edges while smoothing)
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Apply median blur for salt & pepper noise
        denoised = cv2.medianBlur(denoised, 3)
        
        return denoised
    
    def sharpen_image(self, image):
        """
        Sharpen image to enhance facial features
        """
        if image is None:
            return None
        
        # Unsharp masking
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        
        # Alternative: Laplacian sharpening
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        sharpened_laplacian = cv2.filter2D(image, -1, kernel)
        
        # Return the better result (can choose based on image)
        return sharpened
    
    def correct_illumination(self, image):
        """
        Correct uneven illumination
        """
        if image is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply morphological operations to get background illumination
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Subtract background
        corrected = cv2.divide(gray, background, scale=255)
        
        # Convert back to color
        corrected_color = cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)
        
        return corrected_color
    
    def auto_rotate_face(self, image, face_landmarks):
        """
        Automatically rotate face to standard orientation
        """
        if face_landmarks is None:
            return image
        
        # Get eye positions
        left_eye = face_landmarks.get('left_eye', [])
        right_eye = face_landmarks.get('right_eye', [])
        
        if not left_eye or not right_eye:
            return image
        
        # Calculate eye centers
        left_center = np.mean(left_eye, axis=0)
        right_center = np.mean(right_eye, axis=0)
        
        # Calculate rotation angle
        dy = right_center[1] - left_center[1]
        dx = right_center[0] - left_center[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Rotate image
        center = ((left_center[0] + right_center[0]) // 2,
                  (left_center[1] + right_center[1]) // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                                  flags=cv2.INTER_CUBIC)
        
        return rotated
    
    def adjust_skin_tone(self, image):
        """
        Normalize skin tone for consistent recognition
        """
        if image is None:
            return None
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Adjust saturation and value
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.1, 0, 255).astype(np.uint8)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255).astype(np.uint8)
        
        # Convert back
        adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return adjusted
    
    def super_resolution(self, image, scale_factor=2):
        """
        Simple super-resolution to enhance low-quality images
        """
        if image is None:
            return None
        
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        # Use different interpolation methods
        # For faces, Lanczos interpolation gives best results
        enhanced = cv2.resize(image, (new_w, new_h), 
                              interpolation=cv2.INTER_LANCZOS4)
        
        return enhanced
    
    def histogram_equalization(self, image):
        """
        Apply histogram equalization to improve contrast
        """
        if image is None:
            return None
        
        # Convert to YUV
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        
        # Equalize Y channel
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        
        # Convert back
        equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        return equalized
    
    def preprocess_face(self, face_image, enhance_level='medium'):
        """
        Complete preprocessing pipeline for a face image
        
        Args:
            face_image: Cropped face image
            enhance_level: 'basic', 'medium', or 'high'
        
        Returns:
            Preprocessed face image
        """
        if face_image is None or face_image.size == 0:
            return None
        
        # Make a copy
        processed = face_image.copy()
        
        if enhance_level == 'basic':
            # Basic preprocessing
            processed = self.histogram_equalization(processed)
            processed = self.remove_noise(processed)
            
        elif enhance_level == 'medium':
            # Medium preprocessing
            processed = self.enhance_lighting(processed)
            processed = self.histogram_equalization(processed)
            processed = self.remove_noise(processed)
            processed = self.sharpen_image(processed)
            
        elif enhance_level == 'high':
            # Advanced preprocessing
            processed = self.enhance_lighting(processed)
            processed = self.histogram_equalization(processed)
            processed = self.correct_illumination(processed)
            processed = self.remove_noise(processed)
            processed = self.sharpen_image(processed)
            processed = self.adjust_skin_tone(processed)
        
        return processed
    
    def preprocess_frame(self, frame, face_location=None, enhance_level='medium'):
        """
        Preprocess entire frame or specific face region
        """
        if frame is None:
            return None
        
        if face_location:
            # Crop face region
            top, right, bottom, left = face_location
            face = frame[top:bottom, left:right]
            
            # Preprocess face
            processed_face = self.preprocess_face(face, enhance_level)
            
            # Place back in frame
            frame[top:bottom, left:right] = processed_face
            return frame
        else:
            # Preprocess entire frame
            return self.preprocess_face(frame, enhance_level)

# Simple test function
def test_image_processing():
    """Test all preprocessing techniques with webcam"""
    import cv2
    import face_recognition
    
    print("=" * 60)
    print("IMAGE PROCESSING TEST")
    print("=" * 60)
    
    processor = ImagePreprocessor()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Camera not found!")
        return
    
    enhance_level = 'medium'
    print("\nControls:")
    print("  'q' - Quit")
    print("  '1' - Basic enhancement")
    print("  '2' - Medium enhancement")
    print("  '3' - High enhancement")
    print("  '0' - No enhancement")
    print("\nStarting...\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed = processor.preprocess_frame(frame, enhance_level=enhance_level)
        
        # Detect faces to show preprocessing effect
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)
        
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(processed, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Show info
        cv2.putText(processed, f"Enhancement: {enhance_level}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(processed, "Press 1-3 for different levels, 0 for none, q to quit",
                   (10, processed.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('AFRAS - Image Processing Test', processed)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            enhance_level = 'basic'
            print("Basic enhancement")
        elif key == ord('2'):
            enhance_level = 'medium'
            print("Medium enhancement")
        elif key == ord('3'):
            enhance_level = 'high'
            print("High enhancement")
        elif key == ord('0'):
            enhance_level = None
            print("No enhancement")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_image_processing()