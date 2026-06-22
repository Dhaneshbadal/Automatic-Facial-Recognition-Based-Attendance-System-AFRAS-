# recognition/test_model.py
import cv2
import time
import argparse
from pathlib import Path
from recognizer import identify_with_frame, get_registered_students, reload_model

# Simple FPS counter
class FPSMeter:
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
    
    def update(self):
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()
        return self.fps

def start_verification(threshold=0.5, detection_model="hog", resize_factor=0.25):
    """
    Enhanced live verification system with better UI and performance
    
    Args:
        threshold: Recognition threshold (0.3-0.7)
        detection_model: "hog" (fast) or "cnn" (accurate)
        resize_factor: Resize factor for speed
    """
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("❌ Error: Could not open camera.")
        return
    
    # Set camera properties for better performance
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    video_capture.set(cv2.CAP_PROP_FPS, 30)
    
    print("=" * 60)
    print("AFRAS - Live Verification System")
    print("=" * 60)
    print(f"Detection Model: {detection_model}")
    print(f"Recognition Threshold: {threshold}")
    print(f"Resize Factor: {resize_factor}")
    print("\nControls:")
    print("  'q' - Quit")
    print("  'r' - Reload Model")
    print("  's' - Show Registered Students")
    print("\nStarting verification...\n")
    
    fps_meter = FPSMeter()
    frame_count = 0
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
        
        frame_count += 1
        
        # Get the match data and the frame coordinates
        identities = identify_with_frame(
            frame, 
            threshold=threshold,
            resize_factor=resize_factor,
            detection_model=detection_model
        )
        
        # Draw results on frame
        for person in identities:
            top, right, bottom, left = person["coords"]
            name = person["name"]
            confidence = person["confidence"]
            
            # Choose color based on confidence
            if person["is_known"]:
                if confidence > 0.8:
                    color = (0, 255, 0)      # Green - high confidence
                elif confidence > 0.6:
                    color = (0, 255, 255)    # Yellow - medium confidence
                else:
                    color = (0, 165, 255)    # Orange - low confidence
            else:
                color = (0, 0, 255)          # Red - unknown
            
            # Draw rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Create label with confidence
            if person["is_known"]:
                label = f"{name} ({confidence*100:.1f}%)"
            else:
                label = "Unknown"
            
            # Draw background for text
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1
            )
            cv2.rectangle(frame, (left, bottom - 35), 
                         (left + label_width + 10, bottom), color, cv2.FILLED)
            
            # Draw text
            cv2.putText(frame, label, (left + 5, bottom - 8), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
        
        # Display FPS
        fps = fps_meter.update()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display instruction
        cv2.putText(frame, "Press 'q' to quit | 'r' reload | 's' students", 
                   (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show frame
        cv2.imshow('AFRAS - Verification System', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
        
        elif key == ord('r'):
            print("\n🔄 Reloading model...")
            reload_model()
            print("✅ Model reloaded")
        
        elif key == ord('s'):
            print("\n--- Registered Students ---")
            students = get_registered_students()
            if students:
                for s in sorted(students):
                    print(f"  - {s}")
                print(f"Total: {len(students)} students")
            else:
                print("  No students registered yet")
            print("----------------------------\n")
    
    # Cleanup
    video_capture.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total frames processed: {frame_count}")
    print(f"Students in database: {len(get_registered_students())}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AFRAS Live Verification")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Recognition threshold (0.3-0.7, lower = stricter)")
    parser.add_argument("--model", choices=["hog", "cnn"], default="hog",
                        help="Face detection model (hog=faster, cnn=more accurate)")
    parser.add_argument("--resize", type=float, default=0.25,
                        help="Resize factor for speed (0.25 = 4x faster)")
    
    args = parser.parse_args()
    
    start_verification(
        threshold=args.threshold,
        detection_model=args.model,
        resize_factor=args.resize
    )