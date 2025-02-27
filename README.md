# Screen Object Detector

![GitHub stars](https://img.shields.io/github/stars/DJDSTr01/Yolov5-ObjectDetector?style=social)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.7%2B-blue)

A high-performance, real-time object detection system that works directly on your screen. This application uses YOLOv5 to detect objects displayed on your screen and creates smooth, professional overlays to track and label them.



## Features

- **Real-time Detection**: Identifies objects on your screen using state-of-the-art YOLOv5 models
- **Smooth Tracking**: Advanced tracking algorithm with velocity prediction and position smoothing
- **Professional Visualization**: Clean overlay with bounding boxes and labels
- **High Performance**: Optimized for minimal CPU/GPU usage while maintaining smooth operation
- **Easy to Use**: Simple interface, just run and press 'Q' to quit

## ⚠️ Hardware Requirements

> **IMPORTANT**: This application requires a dedicated GPU with sufficient VRAM for smooth performance.

- **GPU**: A modern NVIDIA GPU with at least 4GB VRAM is strongly recommended
  - GTX 1060 6GB or better for optimal performance
  - Performance on older or lower-end GPUs will be significantly degraded
- **CPU**: Intel Core i5/AMD Ryzen 5 or better
- **RAM**: Minimum 8GB, recommended 16GB
- **Operating System**: Windows 10/11 (required for overlay functionality)

**Note**: Running this application on systems with integrated graphics or low-end GPUs will result in:
- Slow detection rates
- Flickering or stuttering overlays
- Inconsistent tracking
- High system resource usage

These issues are hardware limitations and not bugs in the code. For the best experience, ensure your system meets the recommended requirements.

## Requirements

- Python 3.7+
- CUDA-compatible GPU (strongly recommended for optimal performance)
- Windows OS (required for win32gui overlay functionality)

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/DJDSTr01/Yolov5-ObjectDetector.git
   cd Yolov5-ObjectDetector
   ```

2. Create a virtual environment (recommended):
   ```sh
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

4. Install PyTorch with CUDA support (if not automatically installed):
   ```sh
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
   ```

### **Alternative Installation Method**  
Instead of manually installing the dependencies, you can run:  
   ```sh
   installer.bat
   ```  
This will handle everything automatically.


## Usage

Run the main script to start the detector:

```
python screen_detector.py
```

The application will:
1. Load the YOLOv5 model (downloads automatically on first run)
2. Begin detecting and tracking objects on your screen
3. Draw labeled bounding boxes around detected objects

Press `Q` at any time to quit the application.

## Configuration

You can modify the following parameters in the script:

- `confidence`: Detection confidence threshold (default: 0.5)
- `model_size`: YOLOv5 model size ('yolov5s', 'yolov5m', 'yolov5l', or 'yolov5x')
- `detection_interval`: How often to run detection (in seconds)
- `tracking_timeout`: How long to track objects after they disappear

Example:
```python
detector = ScreenObjectDetector(confidence=0.6, model_size='yolov5m')
```

## How It Works

The Screen Object Detector works by:

1. **Screen Capture**: Continuously captures your screen using the MSS library
2. **Object Detection**: Periodically runs YOLOv5 on the captured frames
3. **Object Tracking**: Matches new detections with previously tracked objects
4. **Trajectory Prediction**: Uses velocity-based prediction to smooth object movement
5. **Visualization**: Draws bounding boxes and labels using a transparent Windows overlay

## Use Cases

- **Development Testing**: Test computer vision applications without external cameras
- **Screen Content Analysis**: Identify and track objects in videos and applications
- **Game Analysis**: Track game elements and characters
- **Accessibility Tools**: Highlight important objects on screen

## Performance Tips

- Use a smaller model ('yolov5s') for higher FPS on less powerful hardware
- Increase `detection_interval` to reduce GPU usage (though this may affect tracking quality)
- Run at a lower screen resolution for better performance
- Close other GPU-intensive applications while running the detector
- Consider running on a secondary monitor if your primary display is high-resolution (4K)
- If experiencing flickering, try adjusting the `tracking_timeout` to a higher value

## Troubleshooting

### Common Issues

- **Slow or stuttering detection**: This is typically due to insufficient GPU performance. Try using a smaller model or increase the detection interval.
- **Flickering overlays**: May occur on systems with integrated GPUs or when the system is under heavy load. 
- **High CPU usage**: Normal for screen capture operations, but can be reduced by increasing the sleep time in the main loop.
- **Objects not being detected**: Try lowering the confidence threshold or using a larger model (at the cost of performance).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [MSS Screen Capture](https://github.com/BoboTiG/python-mss)
- [PyWin32](https://github.com/mhammond/pywin32)
