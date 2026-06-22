# recognition/train_model.py
import face_recognition
from image_processor import ImagePreprocessor
import pickle
import cv2
import numpy as np
import os
import sys
from pathlib import Path


def augment_image(image):
    """
    Data augmentation for better model robustness
    Returns list of augmented images
    """
    augmented = []

    # Original
    augmented.append(image.copy())

    # Horizontal flip
    augmented.append(cv2.flip(image, 1))

    # Slight rotations
    rows, cols = image.shape[:2]
    for angle in [-5, 5]:
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated = cv2.warpAffine(image, M, (cols, rows))
        augmented.append(rotated)

    # Brightness adjustment
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for factor in [0.85, 1.15]:
        hsv_adjusted = hsv.copy()
        hsv_adjusted[:, :, 2] = np.clip(hsv_adjusted[:, :, 2] * factor, 0, 255).astype(
            np.uint8
        )
        augmented.append(cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR))

    return augmented


def train_model(
    dataset_path="recognition/dataset/",
    model_path="recognition/models/trained_faces.pkl",
    use_augmentation=True,
    min_encodings_per_student=3,
    enhance_images=True,
    force_retrain=False,
):
    """
    Enhanced training with incremental learning (only trains new students)

    Args:
        dataset_path: Path to dataset folder with student subfolders
        model_path: Path to save trained model
        use_augmentation: Whether to apply data augmentation
        min_encodings_per_student: Minimum encodings required per student
        enhance_images: Whether to enhance images before detection
        force_retrain: If True, retrain all students (ignore existing model)
    """

    dataset_path = Path(dataset_path)
    model_path = Path(model_path)

    # Create model directory if it doesn't exist
    model_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AFRAS - Face Recognition Training (Incremental)")
    print("=" * 60)

    # 1. LOAD EXISTING MODEL (if exists and not force retrain)
    known_encodings = []
    known_names = []
    already_trained_students = set()

    if not force_retrain and model_path.exists():
        print(f"\n📂 Loading existing model from {model_path}")
        try:
            with open(model_path, "rb") as f:
                data = pickle.loads(f.read())
                known_encodings = data.get("encodings", [])
                known_names = data.get("names", [])
                already_trained_students = set(known_names)
            print(f"   ✅ Loaded {len(already_trained_students)} existing student(s)")
            print(f"   📊 Total encodings: {len(known_encodings)}")
        except Exception as e:
            print(f"   ⚠️ Error loading model: {e}")
            print(f"   Starting fresh...")
            already_trained_students = set()
    else:
        if force_retrain:
            print(f"\n🔄 Force retrain mode - ignoring existing model")
        else:
            print(f"\n📂 No existing model found, starting fresh")

    # 2. CHECK DATASET
    if not dataset_path.exists():
        print(f"\n❌ Dataset folder not found: {dataset_path}")
        print("   Please create the folder and add student images.")
        return

    # 3. GET ALL STUDENTS FROM DATASET
    all_students = []
    for student_dir in dataset_path.iterdir():
        if student_dir.is_dir():
            all_students.append(student_dir.name)

    if not all_students:
        print(f"\n❌ No student folders found in {dataset_path}")
        print("   Create folders like: recognition/dataset/student_name/")
        return

    # 4. IDENTIFY NEW STUDENTS (not in model)
    new_students = []
    existing_students = []

    for student in all_students:
        if student in already_trained_students:
            existing_students.append(student)
        else:
            new_students.append(student)

    print(f"\n📊 Dataset Summary:")
    print(f"   Total students in dataset: {len(all_students)}")
    print(f"   Already in model: {len(existing_students)}")
    print(f"   New students to train: {len(new_students)}")

    if new_students:
        print(f"\n   New students found:")
        for s in new_students:
            print(f"     - {s}")

    if existing_students:
        print(f"\n   Existing students (skipping):")
        for s in existing_students[:5]:  # Show first 5
            print(f"     - {s}")
        if len(existing_students) > 5:
            print(f"     ... and {len(existing_students)-5} more")

    # 5. IF NO NEW STUDENTS, EXIT
    if not new_students and not force_retrain:
        print(f"\n✅ No new students found. Model is up to date!")
        print(f"   Total students: {len(already_trained_students)}")
        print(f"   Total encodings: {len(known_encodings)}")
        return

    # 6. TRAIN ONLY NEW STUDENTS
    print(f"\n🎯 Training {len(new_students)} new student(s)...")
    print("-" * 60)

    successful_students = []
    failed_students = []

    # If force retrain, start fresh
    if force_retrain:
        known_encodings = []
        known_names = []
        students_to_train = all_students
        print("\n🔄 Force retrain mode: Training ALL students")
    else:
        students_to_train = new_students

    # Process each student
    for student_name in students_to_train:
        student_dir = dataset_path / student_name
        student_encodings = []

        # Get all image files
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            image_files.extend(student_dir.glob(ext))

        if not image_files:
            print(f"\n⚠️ {student_name}: No images found")
            failed_students.append(student_name)
            continue

        print(f"\n📸 {student_name}: Processing {len(image_files)} image(s)...")

        # Process each image
        for image_path in image_files:
            try:
                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    print(f"   ⚠️ Could not read: {image_path.name}")
                    continue

                # Initialize preprocessor (only once per student)
                if "preprocessor" not in locals():
                    preprocessor = ImagePreprocessor()

                # Apply preprocessing using ImagePreprocessor
                if enhance_images:
                    # Use medium enhancement for better training
                    enhanced = preprocessor.preprocess_face(
                        image, enhance_level="medium"
                    )
                    if enhanced is not None:
                        image = enhanced
                else:
                    # Basic enhancement only
                    enhanced = preprocessor.histogram_equalization(image)
                    if enhanced is not None:
                        image = enhanced

                # Convert to RGB for face_recognition
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Get face encodings
                encodings = face_recognition.face_encodings(rgb_image)

                if len(encodings) > 0:
                    # Use first detected face
                    student_encodings.append(encodings[0])
                    print(f"   ✓ Processed: {image_path.name}")

                    # Data augmentation (skip if we already have enough)
                    if (
                        use_augmentation
                        and len(student_encodings) < min_encodings_per_student * 2
                    ):
                        augmented_images = augment_image(image)
                        for aug_img in augmented_images[:2]:  # Only 2 augmentations
                            rgb_aug = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
                            aug_encodings = face_recognition.face_encodings(rgb_aug)
                            if aug_encodings:
                                student_encodings.append(aug_encodings[0])
                else:
                    print(f"   ⚠️ No face detected in: {image_path.name}")

            except Exception as e:
                print(f"   ❌ Error processing {image_path.name}: {e}")

        # Check if we got enough encodings
        if len(student_encodings) >= min_encodings_per_student:
            # Add all encodings for this student
            known_encodings.extend(student_encodings)
            known_names.extend([student_name] * len(student_encodings))
            successful_students.append(student_name)
            print(f"   ✅ Added {len(student_encodings)} encoding(s)")
        else:
            failed_students.append(student_name)
            print(
                f"   ❌ Only {len(student_encodings)} encoding(s) found (need {min_encodings_per_student})"
            )

    # 7. SAVE MODEL
    if successful_students or force_retrain:
        data = {"encodings": known_encodings, "names": known_names}
        with open(model_path, "wb") as f:
            f.write(pickle.dumps(data))

        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE!")
        print("=" * 60)
        print(f"   Total students in model: {len(set(known_names))}")
        print(f"   Total encodings: {len(known_encodings)}")
        print(f"   Model saved to: {model_path}")

        if successful_students:
            print(f"\n   ✅ Successfully trained new students:")
            for s in successful_students:
                print(f"     ✓ {s}")

        if failed_students:
            print(f"\n   ❌ Failed students (insufficient faces):")
            for s in failed_students:
                print(f"     ✗ {s}")
    else:
        print("\n❌ No students were successfully trained.")
        print("   Make sure each student folder has at least 3 clear face images.")
        print("\n   Tips for better face detection:")
        print("   - Use front-facing, well-lit photos")
        print("   - Make sure faces are at least 200x200 pixels")
        print("   - Avoid blurry or side-profile images")
        print("   - Try with your own photos instead of downloaded datasets")

    print("=" * 60)


def verify_model(model_path="recognition/models/trained_faces.pkl"):
    """Verify that model loaded correctly"""
    model_path = Path(model_path)

    print("\n" + "=" * 60)
    print("MODEL VERIFICATION")
    print("=" * 60)

    try:
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            return False

        with open(model_path, "rb") as f:
            data = pickle.loads(f.read())

        names = data.get("names", [])
        encodings = data.get("encodings", [])

        unique_names = set(names)

        print(f"✅ Model file exists: {model_path}")
        print(f"   File size: {model_path.stat().st_size / 1024:.2f} KB")
        print(f"   Students: {len(unique_names)}")
        print(f"   Total encodings: {len(encodings)}")

        if unique_names:
            print(f"\n   Registered students:")
            for name in sorted(unique_names):
                count = names.count(name)
                print(f"     - {name}: {count} encoding(s)")

        return True

    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        return False


def list_dataset(dataset_path="recognition/dataset/"):
    """List all images in dataset folder"""
    dataset_path = Path(dataset_path)

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    if not dataset_path.exists():
        print(f"❌ Dataset folder not found: {dataset_path}")
        return

    total_images = 0
    total_students = 0
    bad_images = []

    for student_dir in dataset_path.iterdir():
        if student_dir.is_dir():
            image_count = 0
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
                images = list(student_dir.glob(ext))
                image_count += len(images)

                # Quick check for face detection
                for img_path in images[:2]:  # Test first 2 images
                    image = cv2.imread(str(img_path))
                    if image is not None:
                        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        faces = face_recognition.face_locations(rgb)
                        if not faces:
                            bad_images.append(str(img_path))

            total_images += image_count
            total_students += 1
            print(f"   {student_dir.name}: {image_count} image(s)")

    print(f"\n   Total students: {total_students}")
    print(f"   Total images: {total_images}")

    if bad_images:
        print(f"\n   ⚠️ Images with no detectable faces:")
        for img in bad_images[:5]:  # Show first 5
            print(f"     - {img}")
        if len(bad_images) > 5:
            print(f"     ... and {len(bad_images)-5} more")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AFRAS Training Script")
    parser.add_argument(
        "--dataset", default="recognition/dataset/", help="Path to dataset folder"
    )
    parser.add_argument(
        "--output",
        default="recognition/models/trained_faces.pkl",
        help="Path to save model",
    )
    parser.add_argument(
        "--no-augmentation", action="store_true", help="Disable data augmentation"
    )
    parser.add_argument(
        "--min-encodings",
        type=int,
        default=3,
        help="Minimum encodings required per student",
    )
    parser.add_argument(
        "--no-enhance", action="store_true", help="Disable image enhancement"
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retrain ALL students (ignore existing model)",
    )
    parser.add_argument("--verify", action="store_true", help="Verify existing model")
    parser.add_argument("--list", action="store_true", help="List dataset contents")

    args = parser.parse_args()

    if args.verify:
        verify_model(args.output)
    elif args.list:
        list_dataset(args.dataset)
    else:
        train_model(
            dataset_path=args.dataset,
            model_path=args.output,
            use_augmentation=not args.no_augmentation,
            min_encodings_per_student=args.min_encodings,
            enhance_images=not args.no_enhance,
            force_retrain=args.force_retrain,
        )
