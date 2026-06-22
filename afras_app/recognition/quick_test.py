"""
Quick test with your own photos
Take photos of yourself and test immediately
"""
import cv2
import face_recognition
import numpy as np

print("=" * 60)
print("QUICK FACE DETECTION TEST")
print("=" * 60)

# Take a photo
cap = cv2.VideoCapture(0)
print("\n📸 Press SPACE to take a photo, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('Press SPACE to take photo', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        # Save photo
        cv2.imwrite('test_photo.jpg', frame)
        print("\n✅ Photo saved as 'test_photo.jpg'")
        break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

# Test the photo
print("\n🔍 Testing face detection...")
image = cv2.imread('test_photo.jpg')
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

faces = face_recognition.face_locations(rgb)
encodings = face_recognition.face_encodings(rgb, faces)

if faces:
    print(f"✅ Found {len(faces)} face(s)!")
    
    # Draw boxes
    for (top, right, bottom, left) in faces:
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    
    cv2.imwrite('test_result.jpg', image)
    print("✅ Result saved as 'test_result.jpg'")
    print("\nNow you can use this photo for training!")
else:
    print("❌ No face detected. Try:")
    print("  - Better lighting")
    print("  - Face the camera directly")
    print("  - Move closer to camera")  