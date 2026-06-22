#!/usr/bin/env python3
"""
Quick Start Example - Face Recognition Attendance System
This script demonstrates a complete workflow from registration to attendance
"""

from face_recognition_attendance import FaceRecognitionAttendanceSystem
import sys
import os

def quick_start_demo():
    """
    Quick demonstration of the system
    """
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║   FACE RECOGNITION ATTENDANCE SYSTEM - QUICK START DEMO      ║
    ╚══════════════════════════════════════════════════════════════╝
    
    This demo will guide you through:
    1. System initialization
    2. Student registration
    3. Running an attendance session
    4. Viewing results
    
    NOTE: Make sure you have:
    - Webcam connected
    - Good lighting
    - Model files downloaded (.dat files)
    """)
    
    input("Press ENTER to continue...")
    
    # Initialize system
    print("\n" + "="*60)
    print("STEP 1: Initializing System")
    print("="*60)
    
    system = FaceRecognitionAttendanceSystem(
        face_recognition_threshold=0.6,
        attendance_threshold=0.85,
        tracking_interval=1
    )
    
    print("✓ System initialized successfully!")
    print(f"  - Recognition threshold: 0.6")
    print(f"  - Attendance threshold: 85%")
    print(f"  - Tracking interval: 1 second")
    
    # Student registration
    print("\n" + "="*60)
    print("STEP 2: Student Registration")
    print("="*60)
    
    print("\nDo you want to register a new student?")
    print("1. Yes - Register using webcam")
    print("2. Yes - Register from existing images")
    print("3. No - Skip to attendance (use existing registrations)")
    
    choice = input("\nEnter choice (1-3): ")
    
    if choice == "1":
        student_id = input("Enter Student ID (e.g., BCE001): ")
        name = input("Enter Student Name: ")
        
        print(f"\nRegistering {name} ({student_id})...")
        print("The webcam will open. Press SPACE to capture each image.")
        print("You need to capture 5 images from different angles.")
        
        input("Press ENTER when ready...")
        
        # Import registration module
        from student_registration import capture_student_images
        
        image_paths = capture_student_images(student_id, name, num_images=5)
        
        if len(image_paths) >= 3:
            system.register_student(student_id, name, image_paths)
            print(f"✓ Successfully registered {name}!")
        else:
            print("✗ Registration failed - not enough images captured")
            return
    
    elif choice == "2":
        student_id = input("Enter Student ID: ")
        name = input("Enter Student Name: ")
        image_dir = input("Enter image directory path: ")
        
        if not os.path.exists(image_dir):
            print(f"✗ Error: Directory {image_dir} not found")
            return
        
        from student_registration import register_from_directory
        register_from_directory(system, student_id, name, image_dir)
    
    else:
        print("\nSkipping registration - using existing database")
        system.load_registered_faces()
        
        if len(system.registered_faces) == 0:
            print("✗ Error: No registered students found!")
            print("Please register at least one student first.")
            return
        
        print(f"✓ Loaded {len(system.registered_faces)} registered students")
    
    # Run attendance session
    print("\n" + "="*60)
    print("STEP 3: Running Attendance Session")
    print("="*60)
    
    print("\nHow long is your class session?")
    duration = input("Enter duration in minutes (e.g., 60): ")
    
    try:
        duration = int(duration)
    except ValueError:
        print("Invalid duration. Using 5 minutes for demo.")
        duration = 5
    
    print(f"\n✓ Starting {duration}-minute attendance session...")
    print("\nInstructions:")
    print("  - Face the camera")
    print("  - Green box = Recognized student")
    print("  - Red box = Unknown person")
    print("  - Press 'q' to end session early")
    
    input("\nPress ENTER to start session...")
    
    # Run session
    system.run_live_attendance(duration_minutes=duration, camera_index=0)
    
    # View results
    print("\n" + "="*60)
    print("STEP 4: Session Complete!")
    print("="*60)
    
    print("\nAttendance records have been saved to the database.")
    print("\nTo view detailed reports, you can query the database:")
    print("\n  import sqlite3")
    print("  conn = sqlite3.connect('attendance_system.db')")
    print("  cursor = conn.cursor()")
    print("  cursor.execute('SELECT * FROM attendance')")
    print("  for row in cursor.fetchall():")
    print("      print(row)")
    
    print("\n\n✓ Demo completed successfully!")


def run_simple_session():
    """
    Simple session without registration - assumes students already registered
    """
    print("\n" + "="*60)
    print("SIMPLE ATTENDANCE SESSION")
    print("="*60)
    
    # Initialize
    system = FaceRecognitionAttendanceSystem()
    
    # Load existing registrations
    system.load_registered_faces()
    
    if len(system.registered_faces) == 0:
        print("\n✗ No registered students found!")
        print("Please register students first using:")
        print("  python student_registration.py capture <ID> <name>")
        return
    
    print(f"\n✓ Loaded {len(system.registered_faces)} students:")
    for student_id in system.registered_faces.keys():
        print(f"  - {student_id}")
    
    # Get duration
    duration_str = input("\nEnter class duration in minutes (default: 60): ")
    
    try:
        duration = int(duration_str) if duration_str else 60
    except ValueError:
        duration = 60
    
    print(f"\nStarting {duration}-minute session...")
    print("Press 'q' to end early\n")
    
    # Run
    system.run_live_attendance(duration_minutes=duration, camera_index=0)


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║         FACE RECOGNITION ATTENDANCE SYSTEM v1.0              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    if len(sys.argv) > 1 and sys.argv[1] == "simple":
        run_simple_session()
    else:
        quick_start_demo()
