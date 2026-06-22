import os
import shutil
import cv2
import face_recognition
from sklearn.datasets import fetch_lfw_people
from pathlib import Path

def has_detectable_face(image_path):
    """Check if image has a detectable face"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return False
        
        # Resize if too small for detection
        h, w = image.shape[:2]
        if h < 100 or w < 100:
            # Too small, upscale
            scale = 150 / min(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb)
        return len(faces) > 0
    except:
        return False

def setup_lfw_dataset(max_people=50, min_images=5):
    """
    Download LFW dataset and filter only images with detectable faces
    
    Args:
        max_people: Maximum number of people to include
        min_images: Minimum images per person required
    """
    print("=" * 60)
    print("AFRAS - Downloading LFW Dataset")
    print("=" * 60)
    
    # 1. Download the dataset
    print("\n📥 Downloading LFW dataset (this may take a moment)...")
    try:
        fetch_lfw_people(min_faces_per_person=min_images, download_if_missing=True)
        print("   ✅ Download complete!")
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        return
    
    # 2. Locate the download directory
    user_home = os.path.expanduser("~")
    lfw_paths = [
        os.path.join(user_home, "scikit_learn_data", "lfw_home", "lfw_funneled"),
        os.path.join(user_home, "scikit_learn_data", "lfw_home", "lfw"),
        os.path.join(user_home, "scikit_learn_data", "lfw_home", "lfw_funneled", "lfw_funneled"),
    ]
    
    lfw_home = None
    for path in lfw_paths:
        if os.path.exists(path):
            lfw_home = path
            break
    
    if not lfw_home:
        print("❌ Could not find the downloaded images.")
        return
    
    print(f"   📁 Found dataset at: {lfw_home}")
    
    # 3. Setup target directory
    target_base_dir = Path("recognition") / "dataset"
    target_base_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔍 Filtering images with detectable faces...")
    print("-" * 60)
    
    # 4. Process each person
    successful = 0
    failed = 0
    total_images_copied = 0
    
    for person_name in sorted(os.listdir(lfw_home)):
        if successful >= max_people:
            break
            
        source_dir = os.path.join(lfw_home, person_name)
        
        if not os.path.isdir(source_dir):
            continue
        
        # Count images with detectable faces
        valid_images = []
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend([f for f in os.listdir(source_dir) if f.lower().endswith(ext)])
        
        print(f"\n📸 Checking {person_name} ({len(image_files)} images)...")
        
        for img_file in image_files:
            img_path = os.path.join(source_dir, img_file)
            if has_detectable_face(img_path):
                valid_images.append(img_file)
                print(f"   ✓ {img_file}")
            else:
                print(f"   ⚠️ Skipped (no face): {img_file}")
        
        # Only include if enough valid images
        if len(valid_images) >= 3:  # Need at least 3 good images
            dest_dir = target_base_dir / person_name.replace(" ", "_")
            dest_dir.mkdir(exist_ok=True)
            
            # Copy only valid images
            for img_file in valid_images[:10]:  # Max 10 images per person
                src = os.path.join(source_dir, img_file)
                dst = dest_dir / img_file
                shutil.copy2(src, dst)
                total_images_copied += 1
            
            successful += 1
            print(f"   ✅ Added {len(valid_images)} image(s) for {person_name}")
        else:
            failed += 1
            print(f"   ❌ Skipped {person_name} (only {len(valid_images)} valid images)")
    
    print("\n" + "=" * 60)
    print("✅ DATASET PREPARATION COMPLETE!")
    print("=" * 60)
    print(f"   Total people added: {successful}")
    print(f"   Total people skipped: {failed}")
    print(f"   Total images copied: {total_images_copied}")
    print(f"   Dataset location: {target_base_dir}")
    print("\n   Next step: Run training")
    print("   python recognition/train_model.py")
    print("=" * 60)

def download_high_quality_dataset():
    """
    Alternative: Download a better quality face dataset
    This uses kaggle dataset or manual download
    """
    print("=" * 60)
    print("High Quality Face Dataset Download")
    print("=" * 60)
    print("\nOption 1: Use your own photos (recommended)")
    print("   Create folders in recognition/dataset/ with your name")
    print("   Add 3-5 clear, front-facing photos")
    
    print("\nOption 2: Download CelebA dataset (larger, better quality)")
    print("   Visit: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset")
    print("   Download and extract to recognition/dataset/")
    
    print("\nOption 3: Use OpenCV face detector to test your own photos")
    print("   python recognition/test_webcam.py")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--high-quality":
        download_high_quality_dataset()
    else:
        print("Downloading LFW dataset (small images, may not work well)")
        print("For better results, use your own photos instead.")
        print("Press Enter to continue or Ctrl+C to cancel...")
        input()
        setup_lfw_dataset(max_people=20, min_images=3)