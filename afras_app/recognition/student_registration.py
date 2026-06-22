"""
Student Registration Module
Handles bulk registration of students with multiple face images
"""

import cv2
import os
from face_recognition_attendance import FaceRecognitionAttendanceSystem
from pathlib import Path

def capture_student_images(student_id, name, num_images=5):
    """
    Capture multiple images of a student for registration
    
    Args:
        student_id: Student ID
        name: Student name
        num_images: Number of images to capture
    """
    # Create directory for student images
    student_dir = Path(f"student_images/{student_id}")
    student_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return []
    
    print(f"\nCapturing images for {name} ({student_id})")
    print(f"Press SPACE to capture image ({num_images} images needed)")
    print("Press 'q' to quit")
    
    image_paths = []
    count = 0
    
    while count < num_images:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Display
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Image {count + 1}/{num_images}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press SPACE to capture", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Student Registration', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            # Save image
            img_path = student_dir / f"{student_id}_{count + 1}.jpg"
            cv2.imwrite(str(img_path), frame)
            image_paths.append(str(img_path))
            count += 1
            print(f"Captured image {count}/{num_images}")
            
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return image_paths


def register_from_directory(system, student_id, name, directory_path):
    """
    Register a student using images from a directory
    
    Args:
        system: FaceRecognitionAttendanceSystem instance
        student_id: Student ID
        name: Student name
        directory_path: Path to directory containing student images
    """
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Error: Directory {directory_path} not found")
        return
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend([str(p) for p in directory.glob(f"*{ext}")])
        image_paths.extend([str(p) for p in directory.glob(f"*{ext.upper()}")])
    
    if len(image_paths) == 0:
        print(f"Error: No images found in {directory_path}")
        return
    
    print(f"Found {len(image_paths)} images for {name} ({student_id})")
    
    # Register
    system.register_student(student_id, name, image_paths)


def bulk_register_students(csv_file):
    """
    Register multiple students from a CSV file
    
    CSV format: student_id,name,image_directory
    Example: BCE001,Dammara Thagunna,student_images/BCE001
    
    Args:
        csv_file: Path to CSV file
    """
    system = FaceRecognitionAttendanceSystem()
    
    import csv
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            student_id = row['student_id']
            name = row['name']
            image_dir = row['image_directory']
            
            print(f"\n{'='*60}")
            print(f"Registering: {name} ({student_id})")
            print('='*60)
            
            register_from_directory(system, student_id, name, image_dir)
    
    print("\n\nBulk registration completed!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python student_registration.py capture <student_id> <name>")
        print("  python student_registration.py register <student_id> <name> <image_directory>")
        print("  python student_registration.py bulk <csv_file>")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "capture":
        if len(sys.argv) < 4:
            print("Error: Missing arguments")
            print("Usage: python student_registration.py capture <student_id> <name>")
            sys.exit(1)
        
        student_id = sys.argv[2]
        name = " ".join(sys.argv[3:])
        
        image_paths = capture_student_images(student_id, name, num_images=5)
        
        if len(image_paths) > 0:
            system = FaceRecognitionAttendanceSystem()
            system.register_student(student_id, name, image_paths)
    
    elif mode == "register":
        if len(sys.argv) < 5:
            print("Error: Missing arguments")
            print("Usage: python student_registration.py register <student_id> <name> <image_directory>")
            sys.exit(1)
        
        student_id = sys.argv[2]
        name = sys.argv[3]
        image_dir = sys.argv[4]
        
        system = FaceRecognitionAttendanceSystem()
        register_from_directory(system, student_id, name, image_dir)
    
    elif mode == "bulk":
        if len(sys.argv) < 3:
            print("Error: Missing CSV file")
            print("Usage: python student_registration.py bulk <csv_file>")
            sys.exit(1)
        
        csv_file = sys.argv[2]
        bulk_register_students(csv_file)
    
    else:
        print(f"Error: Unknown mode '{mode}'")
        sys.exit(1)
