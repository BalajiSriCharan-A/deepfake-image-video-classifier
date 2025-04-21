# DeepFake Image & Video Classifier using GANs

This project detects whether uploaded images or videos are real or fake using deep learning and GAN-based models.

It includes two parts:
- image_detector/ – Detects fake images using a trained GAN model
- video_detector/ – Detects fake videos using a simple web interface

## How to Use

Image Detector:
1. Open terminal or command prompt
2. Navigate to the image_detector folder:
   cd image_detector
3. Install required packages:
   pip install -r requirements.txt
4. Run the program:
   python image_detector.py

Video Detector:
1. Navigate to the video_detector folder:
   cd video_detector
2. Install required packages:
   pip install -r requirements.txt
3. Run the main Python file:
   python detect_video_fake.py
4. Open index.html in your browser for the web interface.

## Technologies Used

- Python
- TensorFlow
- OpenCV
- Numpy
- GANs (DCGAN)
- HTML + CSS

## Project Structure

deepfake-image-video-classifier/
├── image_detector/        # Code, model, and dependencies for image deepfake detection
├── video_detector/        # Code, frontend, and files for video deepfake detection
└── README.md              # Main project description

## Future Improvements

- Combine both tools into a single web dashboard
- Add webcam/live video detection
- Deploy as a cloud application

## Created by

Amalapuram Balaji Sri Charan  
Email: balajisricharan19@gmail.com  
Location: Visakhapatnam, India
