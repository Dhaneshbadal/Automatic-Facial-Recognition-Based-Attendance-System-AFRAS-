"""
Test Script for Face Recognition Attendance System
Demonstrates system functionality with test data
"""

from face_recognition_attendance import FaceRecognitionAttendanceSystem
import cv2
import numpy as np
from datetime import datetime, timedelta

def test_registration():
    """Test student registration functionality"""
    print("\n" + "="*60)
    print("TEST 1: STUDENT REGISTRATION")
    print("="*60)
    
    system = FaceRecognitionAttendanceSystem()
    
    # Note: Replace with actual image paths
    test_students = [
        {
            'student_id': 'BCE001',
            'name': 'Dammara Thagunna',
            'images': ['student_images/BCE001/img1.jpg', 
                      'student_images/BCE001/img2.jpg',
                      'student_images/BCE001/img3.jpg']
        },
        {
            'student_id': 'BCE002',
            'name': 'Dhanesh Badal',
            'images': ['student_images/BCE002/img1.jpg',
                      'student_images/BCE002/img2.jpg',
                      'student_images/BCE002/img3.jpg']
        }
    ]
    
    # Uncomment to actually register (requires images)
    # for student in test_students:
    #     system.register_student(
    #         student['student_id'],
    #         student['name'],
    #         student['images']
    #     )
    
    print("\nTest completed! (Commented out - add images first)")


def test_face_detection():
    """Test face detection on a single image"""
    print("\n" + "="*60)
    print("TEST 2: FACE DETECTION")
    print("="*60)
    
    system = FaceRecognitionAttendanceSystem()
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("\nPress SPACE to test detection, 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Detect faces
        faces = system.detect_faces(frame)
        
        # Draw rectangles
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display count
        cv2.putText(frame, f"Faces Detected: {len(faces)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Face Detection Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nFace detection test completed!")


def test_threshold_comparison():
    """Test different threshold values"""
    print("\n" + "="*60)
    print("TEST 3: THRESHOLD COMPARISON")
    print("="*60)
    
    # Simulate face distances
    test_distances = [0.3, 0.45, 0.55, 0.62, 0.75, 0.9]
    thresholds = [0.5, 0.6, 0.7]
    
    print("\nTest Distances:", test_distances)
    print("\n" + "-"*60)
    
    for threshold in thresholds:
        matches = sum(1 for d in test_distances if d <= threshold)
        print(f"Threshold {threshold}: {matches}/{len(test_distances)} matches")
        print(f"  Matched: {[d for d in test_distances if d <= threshold]}")
        print(f"  Rejected: {[d for d in test_distances if d > threshold]}")
        print()
    
    print("Recommendation: threshold=0.6 provides good balance")


def test_attendance_calculation():
    """Test attendance percentage calculation"""
    print("\n" + "="*60)
    print("TEST 4: ATTENDANCE CALCULATION")
    print("="*60)
    
    # Simulate session tracking
    session_duration = 60  # 60 minutes
    total_seconds = session_duration * 60
    
    test_cases = [
        {
            'name': 'Student A - Always Present',
            'seconds_present': int(total_seconds * 1.0),  # 100%
        },
        {
            'name': 'Student B - Just Enough',
            'seconds_present': int(total_seconds * 0.85),  # 85%
        },
        {
            'name': 'Student C - Just Below',
            'seconds_present': int(total_seconds * 0.84),  # 84%
        },
        {
            'name': 'Student D - Half Present',
            'seconds_present': int(total_seconds * 0.5),  # 50%
        },
        {
            'name': 'Student E - Absent',
            'seconds_present': 0,  # 0%
        }
    ]
    
    print(f"\nSession Duration: {session_duration} minutes ({total_seconds} seconds)")
    print(f"Attendance Threshold: 85%")
    print("\n" + "-"*60)
    
    for case in test_cases:
        percentage = (case['seconds_present'] / total_seconds) * 100
        status = "PRESENT" if percentage >= 85 else "ABSENT"
        minutes = case['seconds_present'] // 60
        seconds = case['seconds_present'] % 60
        
        print(f"{case['name']}:")
        print(f"  Time Present: {minutes}m {seconds}s ({percentage:.1f}%)")
        print(f"  Status: {status}")
        print()


def test_database_operations():
    """Test database read/write operations"""
    print("\n" + "="*60)
    print("TEST 5: DATABASE OPERATIONS")
    print("="*60)
    
    import sqlite3
    
    system = FaceRecognitionAttendanceSystem()
    
    # Check tables
    conn = sqlite3.connect(system.db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table'
    """)
    
    tables = cursor.fetchall()
    print("\nDatabase Tables:")
    for table in tables:
        print(f"  - {table[0]}")
        
        # Count records
        cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
        count = cursor.fetchone()[0]
        print(f"    Records: {count}")
    
    conn.close()
    print("\nDatabase test completed!")


def test_image_preprocessing():
    """Test image preprocessing pipeline"""
    print("\n" + "="*60)
    print("TEST 6: IMAGE PREPROCESSING")
    print("="*60)
    
    system = FaceRecognitionAttendanceSystem()
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("\nShowing: Original vs Preprocessed")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Preprocess
        preprocessed = system.preprocess_image(frame)
        
        # Combine for display
        combined = np.hstack([frame, preprocessed])
        
        cv2.putText(combined, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "Preprocessed", (frame.shape[1] + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Preprocessing Test', combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nPreprocessing test completed!")


def run_all_tests():
    """Run all test cases"""
    print("\n" + "="*70)
    print(" "*15 + "FACE RECOGNITION ATTENDANCE SYSTEM")
    print(" "*25 + "TEST SUITE")
    print("="*70)
    
    tests = [
        ("Student Registration", test_registration),
        ("Face Detection", test_face_detection),
        ("Threshold Comparison", test_threshold_comparison),
        ("Attendance Calculation", test_attendance_calculation),
        ("Database Operations", test_database_operations),
        ("Image Preprocessing", test_image_preprocessing)
    ]
    
    print("\nAvailable Tests:")
    for i, (name, _) in enumerate(tests, 1):
        print(f"{i}. {name}")
    print("0. Run All Tests")
    
    choice = input("\nSelect test to run (0-6): ")
    
    try:
        choice = int(choice)
        
        if choice == 0:
            for name, test_func in tests:
                try:
                    test_func()
                except Exception as e:
                    print(f"\nError in {name}: {e}")
        elif 1 <= choice <= len(tests):
            tests[choice-1][1]()
        else:
            print("Invalid choice!")
    
    except ValueError:
        print("Please enter a number!")
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")


if __name__ == "__main__":
    run_all_tests()
