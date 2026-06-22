"""
Training script that integrates with Django models
Saves face encodings to the database
"""
import os
import sys
import django
from pathlib import Path

# Setup Django
sys.path.append(str(Path(__file__).parent.parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'afras_app.settings')
django.setup()

from django.contrib.auth.models import User
from recognition.models import Student, FaceRegistration
from recognition.custom_face_model import CustomFaceRecognitionModel
import cv2
import face_recognition
import numpy as np

def train_with_database():
    """
    Train model using data from Django database
    """
    print("=" * 60)
    print("Training Custom Model with Database Data")
    print("=" * 60)
    
    # Initialize model
    model = CustomFaceRecognitionModel()
    
    # Get students from database
    students = Student.objects.filter(is_active=True)
    print(f"\n📊 Found {students.count()} active students")
    
    X = []  # Features
    y = []  # Labels
    
    for student in students:
        print(f"\n📸 Processing: {student.name}")
        
        # Get face registrations
        registrations = student.face_registrations.filter(is_valid=True)
        
        if not registrations.exists():
            print(f"   ⚠️ No face registrations found for {student.name}")
            continue
        
        for reg in registrations:
            image_path = Path(reg.image_path)
            if image_path.exists():
                encoding = model.extract_face_encodings(image_path)
                if encoding is not None:
                    X.append(encoding)
                    y.append(student.name)
                    print(f"   ✓ Added encoding from: {image_path.name}")
        
        # Also check dataset folder
        dataset_path = Path(f"recognition/dataset/{student.name}")
        if dataset_path.exists():
            for img in dataset_path.glob("*.jpg"):
                encoding = model.extract_face_encodings(img)
                if encoding is not None:
                    X.append(encoding)
                    y.append(student.name)
                    print(f"   ✓ Added encoding from dataset: {img.name}")
    
    if len(X) < 10:
        print(f"\n❌ Not enough data! Found {len(X)} encodings, need at least 10")
        return
    
    print(f"\n✅ Total encodings collected: {len(X)}")
    print(f"   Unique students: {len(set(y))}")
    
    # Train model
    X_array = np.array(X)
    y_array = np.array(y)
    
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.preprocessing import LabelEncoder
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_array)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_array, y_encoded, test_size=0.2, random_state=42
    )
    
    # Train SVM
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    svm_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = svm_model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    
    print(f"\n📊 Training Results:")
    print(f"   Test Accuracy: {accuracy*100:.2f}%")
    
    # Save model
    import pickle
    model_data = {
        'model': svm_model,
        'label_encoder': label_encoder,
        'accuracy': accuracy
    }
    
    model_path = Path("recognition/models/custom_model.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✅ Model saved to: {model_path}")
    print(f"   Students in model: {len(label_encoder.classes_)}")
    
    return model_data

if __name__ == "__main__":
    train_with_database()