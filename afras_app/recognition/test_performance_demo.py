"""
Simple performance test for FPS measurement
"""
import cv2
import face_recognition
import time

print("=" * 60)
print("PERFORMANCE TEST - FPS MEASUREMENT")
print("=" * 60)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not found!")
    exit()

# Set lower resolution for better FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\nMeasuring FPS for 10 seconds...")
print("Press 'q' to stop early\n")

frame_count = 0
process_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Process every 2nd frame for speed
    if frame_count % 2 == 0:
        process_count += 1
        
        # Resize for faster processing
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        
        # Face detection
        face_locations = face_recognition.face_locations(rgb, model="hog")
        
        # Draw detected faces (scale back)
        for (top, right, bottom, left) in face_locations:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    
    # Calculate FPS
    elapsed = time.time() - start_time
    if elapsed >= 10:
        break
    
    fps = frame_count / elapsed
    
    # Display info
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Time: {elapsed:.1f}s / 10s", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('AFRAS - Performance Test', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Total frames processed: {frame_count}")
print(f"Time: {elapsed:.1f} seconds")
print(f"Average FPS: {frame_count/elapsed:.1f}")
print(f"Recognition runs: {process_count}")
print("=" * 60)

if frame_count/elapsed >= 15:
    print("✅ Target achieved! FPS >= 15")
else:
    print(f"⚠️ Target is 15 FPS. Current: {frame_count/elapsed:.1f} FPS")