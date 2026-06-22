"""
Complete Face Recognition Model with MySQL Database Storage
- Saves individual face encodings to MySQL (matching your Django DB)
- Saves trained classifier to file
- Supports incremental training
"""

import encodings
import threading

import numpy as np
import cv2
import face_recognition
import pickle
import os
import time
import json

# MySQL connector fallback
try:
    import mysql.connector
    from mysql.connector import Error
except ImportError:
    try:
        import pymysql as mysql
        from pymysql import OperationalError as Error
    except ImportError:
        mysql = None
        Error = Exception

from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class CompleteFaceRecognitionModel:
    """
    Face recognition model that stores encodings in MySQL database
    (matches your Django project's database)
    """
    
    def __init__(self, 
                 model_path="models/trained_classifier.pkl",
                 db_host="localhost",
                 db_user="root",
                 db_password="Lalit@98",  # Your MySQL password
                 db_name="afras_db"):  # Your Django database name
        self.model_path = Path(model_path)
        self.db_config = {
            'host': db_host,
            'user': db_user,
            'password': db_password,
            'database': db_name
        }
        self.partial_path = Path("models/import_progress.json")
        self.model = None
        self.label_encoder = None
        self.model_type = None
        self.is_trained = False
        self.training_stats = {}
        
        # Initialize database tables
        self._init_database()
    
    # ==================== DATABASE SETUP ====================
    
    def _get_connection(self):
        """Get MySQL database connection"""
        if mysql is None:
            print("❌ No MySQL connector installed (mysql-connector-python or pymysql required)")
            return None

        try:
            if hasattr(mysql, "connector"):
                conn = mysql.connector.connect(**self.db_config)
            else:
                conn = mysql.connect(**self.db_config)
            return conn
        except Error as e:
            print(f"❌ MySQL connection error: {e}")
            print("   Make sure MySQL is running and credentials are correct")
            return None
    
    def _init_database(self):
        """Initialize MySQL tables for storing face encodings"""
        conn = self._get_connection()
        if not conn:
            return
        
        cursor = conn.cursor()
        
        # Students table (if not exists)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_students (
                id INT AUTO_INCREMENT PRIMARY KEY,
                student_id VARCHAR(100) UNIQUE NOT NULL,
                name VARCHAR(255) NOT NULL,
                roll_number VARCHAR(50),
                department VARCHAR(100),
                year INT,
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
        ''')
        
        # Face encodings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_encodings (
                id INT AUTO_INCREMENT PRIMARY KEY,
                student_id VARCHAR(100) NOT NULL,
                name VARCHAR(255) NOT NULL,
                encoding TEXT NOT NULL,
                source VARCHAR(50) DEFAULT 'upload',
                image_path VARCHAR(500),
                confidence FLOAT DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES face_students(student_id),
                INDEX idx_student_id (student_id),
                INDEX idx_created_at (created_at)
            )
        ''')
        
        # Training history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_training_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                method VARCHAR(50) NOT NULL,
                num_students INT,
                num_encodings INT,
                accuracy FLOAT,
                training_time FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ MySQL tables initialized in database: {self.db_config['database']}")
    
    # ==================== ENCODING EXTRACTION ====================
    
    def extract_face_encoding(self, image, detection_model="hog"):
        """Extract 128-dimensional face encoding from image"""
        try:
            
            if isinstance(image, str):
                image = cv2.imread(image)
                if image is None:
                    return None
        
            # Check if image is valid
            if image is None or image.size == 0:
                return None
        
            # Convert to RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
            # Detect face with try-catch for individual images
            try:
                face_locations = face_recognition.face_locations(rgb, model=detection_model)
            except Exception as e:
                print(f"      ⚠️ Face detection error: {e}")
                return None
        
            if not face_locations:
                return None
        
            try:
                encodings = face_recognition.face_encodings(rgb, face_locations)
            except Exception as e:
                print(f"      ⚠️ Encoding error: {e}")
                return None
        
            if not encodings:
                return None
        
            return encodings[0]
        
        except Exception as e:
            print(f"      ⚠️ Error processing image: {e}")
            return None
        
    
    def extract_multiple_encodings(self, image_path, num_augmentations=3):
        """Extract multiple encodings (original + augmentations)"""
        encodings = []
    
        # Original encoding
        try:
            original_enc = self.extract_face_encoding(str(image_path))
            if original_enc is not None:
                encodings.append(original_enc)
                print(f"      [ORIGINAL]", end="")
        except Exception as e:
            print(f"      ⚠️ Error with original image: {e}")
    
        # Augmentations
        if num_augmentations > 1:
            try:
                image = cv2.imread(str(image_path))
                if image is not None:
                    # Flip
                    try:
                        flipped = cv2.flip(image, 1)
                        flipped_enc = self.extract_face_encoding(flipped)
                        if flipped_enc is not None:
                            encodings.append(flipped_enc)
                            print(f" + [FLIP]", end="")
                    except Exception as e:
                        print(f"      ⚠️ Flip error: {e}")
                
                    # Small rotation (+5 degrees)
                    try:
                        rows, cols = image.shape[:2]
                        M = cv2.getRotationMatrix2D((cols/2, rows/2), 5, 1)
                        rotated = cv2.warpAffine(image, M, (cols, rows))
                        rotated_enc = self.extract_face_encoding(rotated)
                        if rotated_enc is not None:
                            encodings.append(rotated_enc)
                            print(f" + [ROT+5°]", end="")
                    except Exception as e:
                        print(f"      ⚠️ Rotation error: {e}")
                
                    # Rotation -5 degrees
                    try:
                        rows, cols = image.shape[:2]
                        M = cv2.getRotationMatrix2D((cols/2, rows/2), -5, 1)
                        rotated = cv2.warpAffine(image, M, (cols, rows))
                        rotated_enc = self.extract_face_encoding(rotated)
                        if rotated_enc is not None:
                            encodings.append(rotated_enc)
                            print(f" + [ROT-5°]", end="")
                    except Exception as e:
                        print(f"      ⚠️ Rotation error: {e}")
                
                    # Brightness -15% (darker)
                    try:
                        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                        hsv_adj = hsv.copy()
                        hsv_adj[:, :, 2] = np.clip(hsv_adj[:, :, 2] * 0.85, 0, 255).astype(np.uint8)
                        darker = cv2.cvtColor(hsv_adj, cv2.COLOR_HSV2BGR)
                        darker_enc = self.extract_face_encoding(darker)
                        if darker_enc is not None:
                            encodings.append(darker_enc)
                            print(f" + [DARK]", end="")
                    except Exception as e:
                        print(f"      ⚠️ Brightness error: {e}")
                
                    # Brightness +15% (brighter)
                    try:
                        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                        hsv_adj = hsv.copy()
                        hsv_adj[:, :, 2] = np.clip(hsv_adj[:, :, 2] * 1.15, 0, 255).astype(np.uint8)
                        brighter = cv2.cvtColor(hsv_adj, cv2.COLOR_HSV2BGR)
                        brighter_enc = self.extract_face_encoding(brighter)
                        if brighter_enc is not None:
                            encodings.append(brighter_enc)
                            print(f" + [BRIGHT]", end="")
                    except Exception as e:
                        print(f"      ⚠️ Brightness error: {e}")
                
                    print()  # New line after all augmentations
            except Exception as e:
                print(f"      ⚠️ Augmentation error: {e}")
    
        return encodings
    
    # ==================== DATABASE OPERATIONS ====================
    
    def is_image_processed(self, student_id, image_name):
        """Check if an image has already been processed"""
        conn = self._get_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) FROM face_encodings 
            WHERE student_id = %s AND image_path LIKE %s
        ''', (student_id, f'%{image_name}'))
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return count > 0
    
    def save_encoding_to_db(self, student_id, name, encoding, source='upload', image_path=None):
        """Save a single face encoding to MySQL database"""
        conn = self._get_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # Insert or update student
        cursor.execute('''
            INSERT INTO face_students (student_id, name, updated_at)
            VALUES (%s, %s, NOW())
            ON DUPLICATE KEY UPDATE name = %s, updated_at = NOW()
        ''', (student_id, name, name))
        
        # Insert encoding
        encoding_str = ','.join(map(str, encoding.tolist()))
        cursor.execute('''
            INSERT INTO face_encodings (student_id, name, encoding, source, image_path)
            VALUES (%s, %s, %s, %s, %s)
        ''', (student_id, name, encoding_str, source, image_path))
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
    
    def load_all_encodings_from_db(self, active_only=True):
        """Load all face encodings from MySQL database for training"""
        conn = self._get_connection()
        if not conn:
            return np.array([]), np.array([])
        
        cursor = conn.cursor()
        
        if active_only:
            cursor.execute('''
                SELECT fe.name, fe.encoding 
                FROM face_encodings fe
                JOIN face_students s ON fe.student_id = s.student_id
                WHERE s.is_active = 1
                ORDER BY fe.created_at
            ''')
        else:
            cursor.execute('SELECT name, encoding FROM face_encodings ORDER BY created_at')
        
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if not rows:
            print("⚠️ No encodings found in database")
            return np.array([]), np.array([])
        
        encodings = []
        names = []
        for name, encoding_str in rows:
            encoding = np.array([float(x) for x in encoding_str.split(',')])
            encodings.append(encoding)
            names.append(name)
        
        print(f"📊 Loaded {len(encodings)} encodings, {len(set(names))} unique students")
        return np.array(encodings), np.array(names)
    
    def get_students_from_db(self):
        """Get list of all students from MySQL database"""
        conn = self._get_connection()
        if not conn:
            return []
        
        cursor = conn.cursor()
        cursor.execute('''
            SELECT student_id, name, roll_number, department 
            FROM face_students WHERE is_active = 1
            ORDER BY name
        ''')
        students = cursor.fetchall()
        cursor.close()
        conn.close()
        return students
    
    # ==================== PARTIAL PROGRESS ====================
    
    def _save_partial_progress(self, total_encodings, processed_students, processed_images):
        """Save partial progress to a temporary file"""
        partial_data = {
            'total_encodings': total_encodings,
            'processed_students': list(processed_students),
            'processed_images': list(processed_images),
            'timestamp': time.time()
        }
        
        with open(self.partial_path, 'w') as f:
            json.dump(partial_data, f, indent=2)
    
    def _load_partial_progress(self):
        """Load partial progress from temporary file"""
        if not self.partial_path.exists():
            return set(), set(), 0
        
        try:
            with open(self.partial_path, 'r') as f:
                data = json.load(f)
            return (set(data.get('processed_students', [])), 
                    set(data.get('processed_images', [])), 
                    data.get('total_encodings', 0))
        except:
            return set(), set(), 0
    
    def _clear_partial_progress(self):
        """Clear partial progress file"""
        if self.partial_path.exists():
            self.partial_path.unlink()
    
    # ==================== DATASET IMPORT ====================
    
    def import_dataset_to_db(self, dataset_path="recognition/dataset/", use_augmentation=True, resume=True):
        """Import images from folder structure to MySQL database"""
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            print(f"❌ Dataset not found: {dataset_path}")
            return 0
        
        # Load partial progress if resuming
        processed_students, processed_images, total_encodings = self._load_partial_progress()
        
        if resume and processed_students:
            print(f"\n📂 Resuming from previous import...")
            print(f"   Already processed students: {len(processed_students)}")
            print(f"   Already processed images: {len(processed_images)}")
            print(f"   Already saved encodings: {total_encodings}")
        
        all_students = [d for d in dataset_path.iterdir() if d.is_dir()]
        
        print(f"\n📊 Total students to process: {len(all_students)}")
        print("-" * 60)
        
        new_encodings_count = 0
        students_processed = 0
        
        for i, student_dir in enumerate(all_students):
            if not student_dir.is_dir():
                continue
            
            student_name = student_dir.name
            student_id = student_name.replace(' ', '_').lower()
            
            # Skip if already fully processed
            if student_id in processed_students and resume:
                print(f"\n⏭️ [{i+1}/{len(all_students)}] Skipping {student_name} (already imported)")
                continue
            
            print(f"\n📸 [{i+1}/{len(all_students)}] Processing {student_name}:")
            
            images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                images.extend(student_dir.glob(ext))
            
            if not images:
                print(f"   ⚠️ No images found")
                continue
            
            student_encodings = []
            new_images_found = 0
            skipped_images = 0
            
            for img_path in images:
                img_id = f"{student_id}_{img_path.name}"
                
                if img_id in processed_images and resume:
                    print(f"   ⏭️ Skipping {img_path.name} (already processed)")
                    skipped_images += 1
                    continue
                
                if use_augmentation:
                    encodings = self.extract_multiple_encodings(img_path, num_augmentations=3)
                    valid_enc = [enc for enc in encodings if enc is not None]
                    if valid_enc:
                        student_encodings.extend(valid_enc)
                        new_images_found += 1
                        print(f"   ✓ {img_path.name} -> {len(valid_enc)} encoding(s)")
                        processed_images.add(img_id)
                    else:
                        print(f"   ⚠️ No face: {img_path.name}")
                else:
                    enc = self.extract_face_encoding(str(img_path))
                    if enc is not None:
                        student_encodings.append(enc)
                        new_images_found += 1
                        print(f"   ✓ {img_path.name} -> 1 encoding")
                        processed_images.add(img_id)
                    else:
                        print(f"   ⚠️ No face: {img_path.name}")
            
            if new_images_found == 0:
                print(f"   ❌ No new valid images for {student_name}")
                continue
            
            # Save encodings to database
            for enc in student_encodings:
                self.save_encoding_to_db(student_id, student_name, enc, 
                                        source='dataset', image_path=str(img_path))
                new_encodings_count += 1
                total_encodings += 1
            
            processed_students.add(student_id)
            students_processed += 1
            
            print(f"   ✅ Saved {len(student_encodings)} encodings from {new_images_found} new images")
            print(f"   📊 Total so far: {total_encodings} encodings from {len(processed_students)} students")
            
            # Save progress after each student
            self._save_partial_progress(total_encodings, processed_students, processed_images)
        
        # Clear partial file on success
        if students_processed > 0:
            self._clear_partial_progress()
            print(f"\n🗑️ Removed partial progress file")
        
        print(f"\n" + "=" * 60)
        print("✅ IMPORT COMPLETE!")
        print("=" * 60)
        print(f"   New encodings saved: {new_encodings_count}")
        print(f"   Total encodings in DB: {total_encodings}")
        print(f"   Students processed: {students_processed}")
        print("=" * 60)
        
        return total_encodings
    
    # ==================== TRAINING ====================
    
    def train_svm(self, X, y):
        """Train SVM classifier with progress tracking"""
        print("\n🎯 Training SVM Classifier...")
    
        from sklearn.svm import SVC
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from collections import Counter
    
        # Filter out classes with less than 2 samples
        class_counts = Counter(y)
        min_samples_per_class = 2
    
        valid_classes = [cls for cls, count in class_counts.items() if count >= min_samples_per_class]
    
        if len(valid_classes) < len(class_counts):
            print(f"   ⚠️ Filtering out {len(class_counts) - len(valid_classes)} students with < {min_samples_per_class} encoding(s)")
    
        # Filter data
        mask = [label in valid_classes for label in y]
        X_filtered = X[mask]
        y_filtered = y[mask]
    
        if len(X_filtered) == 0:
            print("   ❌ Not enough data after filtering!")
            return 0.0
    
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y_filtered)
    
        # Split data
        if len(X_filtered) > 10 and len(set(y_filtered)) > 1:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_filtered, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                )
            except ValueError:
                print("   ⚠️ Stratified split failed, using regular split...")
                X_train, X_test, y_train, y_test = train_test_split(
                    X_filtered, y_encoded, test_size=0.2, random_state=42
                )
        else:
            X_train, y_train = X_filtered, y_encoded
            X_test, y_test = X_filtered, y_encoded
    
        print(f"   Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        print(f"   Number of classes: {len(set(y_filtered))}")

        # Progress indicator
        stop_progress = threading.Event()
    
        def show_progress():
            dots = 0
            while not stop_progress.is_set():
                dots = (dots + 1) % 4
            print(f"\r   Training SVM in progress...{'.' * dots} (may take 20-60 min)", end="", flush=True)
            time.sleep(3)
    
        progress_thread = threading.Thread(target=show_progress, daemon=True)
        progress_thread.start()
    
        # Train SVM
        start_time = time.time()
    
        self.model = SVC(
            kernel='rbf', 
            C=0.5,
            gamma='scale',
            probability=True, 
            random_state=42,
            class_weight='balanced',
            cache_size=1000,
            max_iter=1000
        )
    
        self.model.fit(X_train, y_train)
    
        # Stop progress indicator
        stop_progress.set()
        progress_thread.join(timeout=1)
    
        training_time = time.time() - start_time
        print(f"\r   ✅ Training completed in {training_time/60:.1f} minutes    ")
    
        # Evaluate
        if len(X_test) > 0:
            print("   Evaluating on test set...")
            y_pred = self.model.predict(X_test)
            accuracy = np.mean(y_pred == y_test)
        else:
            accuracy = 1.0
    
        print(f"   ✅ Test accuracy: {accuracy*100:.2f}%")
        self.model_type = 'svm'
    
        return accuracy
    
    
    
    def train(self, method="svm"):
        """Train model using encodings from database"""
        print("=" * 60)
        print("FACE RECOGNITION MODEL - TRAINING")
        print("=" * 60)
        print(f"Method: {method.upper()}")
        
        X, y = self.load_all_encodings_from_db()
        
        if len(X) == 0:
            print("\n❌ No encodings in database! Import dataset first:")
            print("   python custom_face_model.py --import")
            return False
        
        print(f"\n📊 Training data: {len(X)} encodings, {len(set(y))} students")
        
        start_time = time.time()
        accuracy = self.train_svm(X, y)
        training_time = time.time() - start_time
        
        # Save training history
        conn = self._get_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO face_training_history (method, num_students, num_encodings, accuracy, training_time)
                VALUES (%s, %s, %s, %s, %s)
            ''', (method, len(set(y)), len(X), accuracy, training_time))
            conn.commit()
            cursor.close()
            conn.close()
        
        # Save model
        self.is_trained = True
        self._save_model()
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE!")
        print("=" * 60)
        print(f"   Method: {method.upper()}")
        print(f"   Accuracy: {accuracy*100:.2f}%")
        print(f"   Students: {len(set(y))}")
        print(f"   Encodings: {len(X)}")
        print(f"   Time: {training_time:.2f}s")
        print(f"   Model saved: {self.model_path}")
        print("=" * 60)
        
        return True
    
    def _save_model(self):
        """Save trained classifier to file"""
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            
            model_data = {
                'model': self.model,
                'label_encoder': self.label_encoder,
                'model_type': self.model_type,
                'is_trained': self.is_trained,
                'timestamp': time.time()
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"\n💾 Classifier saved: {self.model_path}")
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
    
    def load_model(self):
        """Load trained classifier from file"""
        if not self.model_path.exists():
            print(f"⚠️ Model not found: {self.model_path}")
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.model_type = model_data['model_type']
            self.is_trained = model_data['is_trained']
            
            print(f"✅ Model loaded: {self.model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    # ==================== RECOGNITION ====================
    
    def predict(self, face_encoding):
        """Predict person from face encoding"""
        if not self.is_trained or self.model is None:
            return "Unknown", 0.0
        
        try:
            encoding = face_encoding.reshape(1, -1)
            
            if self.model_type == 'svm' and hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(encoding)[0]
                prediction = self.model.predict(encoding)[0]
                confidence = np.max(probabilities)
                
                if self.label_encoder:
                    name = self.label_encoder.inverse_transform([prediction])[0]
                else:
                    name = prediction
            else:
                prediction = self.model.predict(encoding)[0]
                confidence = 0.8
                
                if self.label_encoder:
                    name = self.label_encoder.inverse_transform([prediction])[0]
                else:
                    name = prediction
            
            return name, confidence
            
        except Exception as e:
            return "Unknown", 0.0
    
    def predict_from_frame(self, frame, threshold=0.6):
        """Recognize faces in a video frame"""
        if frame is None:
            return []
        
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb, model="hog")
        face_encodings = face_recognition.face_encodings(rgb, face_locations)
        
        results = []
        scale = 4
        
        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            name, confidence = self.predict(encoding)
            
            top *= scale
            right *= scale
            bottom *= scale
            left *= scale
            
            results.append({
                'name': name,
                'confidence': confidence,
                'coords': (top, right, bottom, left),
                'is_known': confidence >= threshold and name != "Unknown"
            })
        
        return results
    
    def get_stats(self):
        """Get database statistics"""
        conn = self._get_connection()
        if not conn:
            return {}
        
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM face_students WHERE is_active = 1")
        student_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM face_encodings")
        encoding_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT image_path) FROM face_encodings")
        image_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT method, accuracy, training_time FROM face_training_history ORDER BY id DESC LIMIT 1")
        last_training = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        return {
            'students': student_count,
            'encodings': encoding_count,
            'images': image_count,
            'last_training_method': last_training[0] if last_training else None,
            'last_training_accuracy': last_training[1] if last_training else None,
            'last_training_time': last_training[2] if last_training else None
        }


# ==================== COMMAND LINE ====================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete Face Recognition Model with MySQL")
    parser.add_argument("--import", dest="import_dataset", action="store_true", 
                       help="Import dataset to MySQL database")
    parser.add_argument("--train", action="store_true", help="Train model from database")
    parser.add_argument("--test", action="store_true", help="Test with webcam")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--method", default="svm", help="Training method (default: svm)")
    parser.add_argument("--dataset", default="recognition/dataset/", help="Dataset path")
    parser.add_argument("--host", default="localhost", help="MySQL host")
    parser.add_argument("--user", default="root", help="MySQL user")
    parser.add_argument("--password", default="", help="MySQL password")
    parser.add_argument("--database", default="afras_db", help="MySQL database name")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    
    args = parser.parse_args()
    
    model = CompleteFaceRecognitionModel(
        model_path="models/trained_classifier.pkl",
        db_host=args.host,
        db_user=args.user,
        db_password=args.password,
        db_name=args.database
    )
    
    if getattr(args, 'import_dataset'):
        model.import_dataset_to_db(args.dataset, use_augmentation=not args.no_augment)
        
    elif args.train:
        model.train(method=args.method)
        
    elif args.test:
        if not model.load_model():
            print("❌ No trained model found. Train first with: --train")
        else:
            test_webcam(model)
    
    elif args.stats:
        stats = model.get_stats()
        print("\n" + "=" * 60)
        print("DATABASE STATISTICS (MySQL)")
        print("=" * 60)
        print(f"   Students: {stats.get('students', 0)}")
        print(f"   Images processed: {stats.get('images', 0)}")
        print(f"   Encodings: {stats.get('encodings', 0)}")
        if stats.get('last_training_accuracy'):
            print(f"   Last Accuracy: {stats['last_training_accuracy']*100:.2f}%")
        print("=" * 60)
    
    else:
        print("=" * 60)
        print("FACE RECOGNITION MODEL with MySQL")
        print("=" * 60)
        print("\nUsage:")
        print("  Import: python custom_face_model.py --import --host localhost --user root --password yourpass --database afras_db")
        print("  Train:  python custom_face_model.py --train")
        print("  Test:   python custom_face_model.py --test")
        print("  Stats:  python custom_face_model.py --stats")
        print("=" * 60)


def test_webcam(model):
    """Test with webcam"""
    import cv2
    
    print("=" * 60)
    print("TESTING WITH WEBCAM")
    print("=" * 60)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not found!")
        return
    
    print("\n✅ Camera ready! Press 'q' to quit\n")
    
    frame_count = 0
    recognized_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % 2 == 0:
            results = model.predict_from_frame(frame, threshold=0.6)
            
            for result in results:
                top, right, bottom, left = result['coords']
                name = result['name']
                confidence = result['confidence']
                
                if result['is_known']:
                    recognized_count += 1
                    color = (0, 255, 0)
                    label = f"{name} ({confidence*100:.1f}%)"
                else:
                    color = (0, 0, 255)
                    label = "Unknown"
                
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, label, (left, top-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.putText(frame, f"Frames: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(frame, f"Recognized: {recognized_count}", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        
        cv2.imshow('Face Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nTotal frames: {frame_count}")
    print(f"Recognized faces: {recognized_count}")


if __name__ == "__main__":
    main()