# recognition/recognizer.py
import face_recognition
import pickle
import numpy as np
import cv2
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model caching
_MODEL_CACHE = None
_MODEL_PATH = "recognition/models/trained_faces.pkl"

def load_model(model_path=None):
    """
    Load the trained model into memory with caching
    """
    global _MODEL_CACHE
    
    if model_path is None:
        model_path = _MODEL_PATH
    
    # Return cached model if available
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    
    # Load model from file
    if not os.path.exists(model_path):
        logger.warning(f"Model file not found at {model_path}")
        return {"encodings": np.array([]), "names": np.array([])}
    
    try:
        with open(model_path, "rb") as f:
            data = pickle.loads(f.read())
            # Convert to numpy arrays for faster computation
            data["encodings"] = np.array(data.get("encodings", []))
            data["names"] = np.array(data.get("names", []))
        _MODEL_CACHE = data
        logger.info(f"Loaded {len(data['names'])} face encodings")
        return data
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return {"encodings": np.array([]), "names": np.array([])}

def reload_model():
    """
    Force reload the model from disk
    """
    global _MODEL_CACHE
    _MODEL_CACHE = None
    return load_model()

def identify_with_frame(frame, threshold=0.5, resize_factor=0.25, 
                        detection_model="hog", encoding_model="large"):
    """
    Enhanced version - Converts live frame to encoding and matches against backend data.
    
    Args:
        frame: BGR image from OpenCV
        threshold: Recognition threshold (0.3-0.7, lower = stricter)
        resize_factor: Factor to resize frame for speed (0.25 = 4x faster)
        detection_model: "hog" (fast) or "cnn" (accurate)
        encoding_model: "small" (fast) or "large" (accurate)
    
    Returns:
        List of dictionaries with name, confidence, and coordinates
    """
    if frame is None:
        return []
    
    # Load model
    data = load_model()
    
    # 1. Prepare the frame (Resize for 15 FPS goal)
    original_height, original_width = frame.shape[:2]
    small_width = int(original_width * resize_factor)
    small_height = int(original_height * resize_factor)
    small_frame = cv2.resize(frame, (small_width, small_height))
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # 2. Find faces in live feed (Enhanced detection)
    face_locations = face_recognition.face_locations(
        rgb_small_frame, 
        model=detection_model
    )
    
    # 3. Get face encodings (Enhanced encoding)
    face_encodings = face_recognition.face_encodings(
        rgb_small_frame, 
        face_locations,
        num_jitters=1,  # Increase for better accuracy (1-10)
        model=encoding_model
    )
    
    # Scale coordinates back to original size
    scale_factor = 1.0 / resize_factor
    
    results = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        confidence = 0.0
        
        # Scale coordinates back
        scaled_top = int(top * scale_factor)
        scaled_right = int(right * scale_factor)
        scaled_bottom = int(bottom * scale_factor)
        scaled_left = int(left * scale_factor)
        
        if len(data["encodings"]) > 0:
            # 4. Calculate Euclidean Distance between live face and ALL saved faces
            face_distances = np.linalg.norm(data["encodings"] - face_encoding, axis=1)
            
            # 5. Find the index of the smallest distance
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]
            
            # 6. Calculate confidence score (0-1)
            confidence = max(0, min(1, 1 - (best_distance / 0.8)))
            
            # 7. Apply the Threshold logic
            if best_distance <= threshold:
                name = data["names"][best_match_index]
        
        results.append({
            "name": name,
            "confidence": confidence,
            "coords": (scaled_top, scaled_right, scaled_bottom, scaled_left),
            "is_known": name != "Unknown"
        })
    
    return results

def identify_student(frame, threshold=0.5):
    """
    Legacy function - Returns only names (backward compatibility)
    """
    results = identify_with_frame(frame, threshold)
    return [r["name"] for r in results]

def get_registered_students():
    """
    Get list of all registered students
    """
    data = load_model()
    return list(set(data["names"])) if len(data["names"]) > 0 else []

def get_student_count():
    """
    Get number of unique students in model
    """
    return len(get_registered_students())