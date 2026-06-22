#!/bin/bash

# Face Recognition Attendance System - Setup Script
# This script sets up the complete environment

echo "============================================================"
echo "Face Recognition Attendance System - Setup"
echo "============================================================"
echo ""

# Check Python version
echo "[1/6] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed!"
    exit 1
fi

# Create virtual environment
echo ""
echo "[2/6] Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "[3/6] Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "[4/6] Installing Python packages..."
pip install opencv-python==4.8.1.78
pip install numpy==1.24.3

# Install dlib (may take time)
echo ""
echo "Installing dlib (this may take 5-10 minutes)..."
pip install dlib==19.24.2

# Install additional packages
pip install Pillow pandas matplotlib

echo ""
echo "[5/6] Downloading pre-trained models..."

# Download shape predictor
if [ ! -f "shape_predictor_68_face_landmarks.dat" ]; then
    echo "Downloading face landmarks model..."
    wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    bunzip2 shape_predictor_68_face_landmarks.dat.bz2
    echo "Face landmarks model downloaded!"
else
    echo "Face landmarks model already exists."
fi

# Download face recognition model
if [ ! -f "dlib_face_recognition_resnet_model_v1.dat" ]; then
    echo "Downloading face recognition model..."
    wget http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
    bunzip2 dlib_face_recognition_resnet_model_v1.dat.bz2
    echo "Face recognition model downloaded!"
else
    echo "Face recognition model already exists."
fi

# Create directories
echo ""
echo "[6/6] Creating project directories..."
mkdir -p student_images
mkdir -p reports
mkdir -p logs

echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Register students:"
echo "   python student_registration.py capture BCE001 \"Student Name\""
echo ""
echo "3. Run attendance session:"
echo "   python -c 'from face_recognition_attendance import FaceRecognitionAttendanceSystem; s=FaceRecognitionAttendanceSystem(); s.run_live_attendance(60)'"
echo ""
echo "4. Run tests:"
echo "   python test_system.py"
echo ""
