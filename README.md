# car-parking-space-counter

## Description
A Python-based real-time parking management system that uses computer vision to detect parked vehicles and count available spaces. Ideal for monitoring parking occupancy in real time using video feeds.

## Features
- Real-time occupancy detection
- Vehicle detection and counting with OpenCV and CVZone
- Visual feedback (occupied vs available)
- Lightweight and easily adaptable to different parking layouts

## Technologies
- Python
- OpenCV
- CVZone
- NumPy

## Usage
1. Define parking space boundaries (e.g., via manual labeling).
2. Run the main script (e.g., `main.py`) with the video feed.
3. View live counts and space status updates.

## Future Enhancements
- Integration with deep-learning detection models (e.g., YOLOv8 + DeepSORT) :contentReference[oaicite:0]{index=0}
- Automated parking boundary detection instead of manual input :contentReference[oaicite:1]{index=1}
- Mobile or web dashboard for real-time monitoring and alerts.

## Acknowledgements
Inspired by similar computer-vision projects and tutorials in the open-source community :contentReference[oaicite:2]{index=2}.
