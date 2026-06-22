"""
Test custom face recognition model with webcam
"""
import cv2
import pickle
import numpy as np
import face_recognition
from pathlib import Path

class CustomModelTester:
    def __init__(self, model_path="recognition/models/custom_model.pkl"):
        self.model = None
        self.label_encoder = None
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load trained model"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            return False
        
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.label_encoder = data['label_encoder']
        
        print(f"✅ Model loaded: {len(self.label_encoder.classes_)} students")
        return True
    
    def predict(self, face_encoding):
        """Predict from face encoding"""
        if self.model is None:
            return "Unknown", 0.0
        
        encoding = face_encoding.reshape(1, -1)
        
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(encoding)[0]
            pred = self.model.predict(encoding)[0]
            confidence = np.max(probs)
        else:
            pred = self.model.predict(encoding)[0]
            confidence = 1.0
        
        name = self.label_encoder.inverse_transform([pred])[0]
        return name, confidence
    
    def run(self):
        """Run webcam test"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Camera not found!")
            return
        
        print("\n🎯 Custom Model Test Running...")
        print("Press 'q' to quit\n")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 2nd frame
            if frame_count % 2 == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb)
                face_encodings = face_recognition.face_encodings(rgb, face_locations)
                
                for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                    name, confidence = self.predict(encoding)
                    
                    # Color: Green for known, Red for unknown
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    
                    # Draw box and label
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    label = f"{name} ({confidence:.2f})" if name != "Unknown" else "Unknown"
                    cv2.putText(frame, label, (left, top-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.imshow('Custom Model Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tester = CustomModelTester()
    tester.run()