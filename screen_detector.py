import torch
import numpy as np
from mss import mss
from PIL import Image
import win32gui
import win32ui
import win32con
import win32api
import time
from collections import deque

class TrackedObject:
    def __init__(self, bbox, class_id, confidence):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.class_id = class_id
        self.confidence = confidence
        self.last_seen = time.time()
        self.track_id = None
        
        # Position history for smooth tracking
        self.position_history = deque(maxlen=8)
        self.position_history.append(bbox)
        
        # Velocity tracking for prediction
        self.velocity = [0, 0, 0, 0]  # vx1, vy1, vx2, vy2
        self.smoothing_factor = 0.6
        
    def get_smoothed_position(self):
        """Get smoothed position using weighted average"""
        if len(self.position_history) < 2:
            return self.bbox
            
        # Apply exponential weighting to position history
        positions = np.array(self.position_history)
        weights = np.exp(np.linspace(-1.5, 0, len(positions)))
        weights /= weights.sum()
        
        smoothed = np.sum(positions * weights[:, np.newaxis], axis=0)
        return smoothed.tolist()
        
    def update_position(self, new_bbox, new_confidence):
        """Update position with smoothing"""
        old_pos = self.get_smoothed_position()
        dt = time.time() - self.last_seen
        
        if dt > 0:
            # Calculate new velocity
            new_velocity = [(new_bbox[i] - old_pos[i]) / dt for i in range(4)]
            
            # Update velocity with smoothing
            for i in range(4):
                self.velocity[i] = (self.smoothing_factor * new_velocity[i] + 
                                   (1 - self.smoothing_factor) * self.velocity[i])
        
        # Update confidence and position history
        self.confidence = new_confidence
        self.position_history.append(new_bbox)
        self.last_seen = time.time()
        
    def predict_position(self, dt):
        """Predict position using velocity"""
        current_pos = self.get_smoothed_position()
        return [current_pos[i] + self.velocity[i] * dt for i in range(4)]

class ScreenObjectDetector:
    def __init__(self, confidence=0.5, model_size='yolov5s'):
        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', model_size, pretrained=True)
        self.model.conf = confidence
        
        # Set up device (GPU if available, otherwise CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Screen capture setup
        self.sct = mss()
        self.screen_width = win32api.GetSystemMetrics(0)
        self.screen_height = win32api.GetSystemMetrics(1)
        self.screen_area = {'top': 0, 'left': 0, 'width': self.screen_width, 'height': self.screen_height}
        
        # Set up overlay window for drawing
        self.setup_overlay()
        
        # Tracking parameters
        self.fps = 0
        self.last_time = time.time()
        self.tracked_objects = {}
        self.next_track_id = 0
        self.detection_interval = 0.05  # Run detection every 50ms
        self.last_detection = 0
        self.tracking_timeout = 0.5  # Remove objects not seen for 0.5 seconds
        self.iou_threshold = 0.4  # Threshold for matching detections to tracks
        
        self.is_cleaned_up = False
 
    def calculate_iou(self, box1, box2):
        """Calculate intersection over union between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

    def update_tracked_objects(self, detections):
        """Match new detections to existing tracked objects"""
        current_time = time.time()
        dt = current_time - self.last_detection
        
        # Predict new positions for existing objects
        for obj in self.tracked_objects.values():
            obj.bbox = obj.predict_position(dt)
        
        # Match detections to existing tracks
        matched_track_ids = set()
        matched_detection_indices = set()
        
        # Match based on IoU
        for track_id, tracked_obj in self.tracked_objects.items():
            best_match_idx = None
            best_iou = self.iou_threshold
            
            for i, det in enumerate(detections):
                if i in matched_detection_indices:
                    continue
                    
                det_bbox = det[:4].tolist()
                iou = self.calculate_iou(tracked_obj.bbox, det_bbox)
                
                if iou > best_iou and tracked_obj.class_id == int(det[5]):
                    best_iou = iou
                    best_match_idx = i
            
            if best_match_idx is not None:
                # Update the matched track
                det = detections[best_match_idx]
                tracked_obj.update_position(det[:4].tolist(), float(det[4]))
                matched_track_ids.add(track_id)
                matched_detection_indices.add(best_match_idx)
        
        # Add new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_detection_indices:
                bbox = det[:4].tolist()
                new_obj = TrackedObject(bbox, int(det[5]), float(det[4]))
                new_obj.track_id = self.next_track_id
                self.tracked_objects[self.next_track_id] = new_obj
                self.next_track_id += 1
        
        # Remove stale tracks
        self.tracked_objects = {
            track_id: obj for track_id, obj in self.tracked_objects.items()
            if (current_time - obj.last_seen <= self.tracking_timeout and 
                track_id in matched_track_ids or current_time - obj.last_seen < 0.1)
        }

    def draw_objects(self):
        """Draw tracked objects on the overlay"""
        try:
            # Clear previous drawings
            self.dc.FillRect((0, 0, self.screen_width, self.screen_height), self.brush)
            
            # Draw each tracked object
            for obj in self.tracked_objects.values():
                # Get smoothed position
                x1, y1, x2, y2 = map(int, obj.get_smoothed_position())
                
                # Draw bounding box
                self.dc.SelectObject(self.pens['tracking'])
                self.dc.MoveTo((x1, y1))
                self.dc.LineTo((x2, y1))
                self.dc.LineTo((x2, y2))
                self.dc.LineTo((x1, y2))
                self.dc.LineTo((x1, y1))
                
                # Draw label with class name and confidence
                self.dc.SelectObject(self.pens['text'])
                label = f"{self.model.names[obj.class_id]} ({obj.confidence:.2f})"
                self.dc.TextOut(x1, y1 - 15, label)
            
            # Draw FPS counter
            self.dc.SelectObject(self.pens['text'])
            self.dc.TextOut(10, 10, f"FPS: {self.fps:.1f}")
            
        except Exception as e:
            print(f"Drawing error: {e}")

    def capture_screen(self):
        """Capture the screen using MSS"""
        screenshot = self.sct.grab(self.screen_area)
        img_rgb = Image.frombytes('RGB', (screenshot.width, screenshot.height), screenshot.rgb)
        return np.array(img_rgb)

    def update_fps(self):
        """Calculate and update FPS"""
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time

    def setup_overlay(self):
        """Set up the transparent overlay for drawing on the screen"""
        self.hwnd = win32gui.GetDesktopWindow()
        self.hdc = win32gui.GetDC(self.hwnd)
        self.dc = win32ui.CreateDCFromHandle(self.hdc)
        
        # Create transparent brush and colored pens
        self.brush = win32ui.CreateBrush(win32con.BS_HOLLOW, 0, 0)
        self.pens = {
            'tracking': win32ui.CreatePen(win32con.PS_SOLID, 2, 0x00FF00),  # Green
            'text': win32ui.CreatePen(win32con.PS_SOLID, 1, 0xFFFF00)       # Yellow
        }

    def cleanup(self):
        """Clean up resources"""
        if not self.is_cleaned_up:
            try:
                if hasattr(self, 'dc'):
                    self.dc.DeleteDC()
                if hasattr(self, 'brush'):
                    self.brush.DeleteObject()
                for pen in getattr(self, 'pens', {}).values():
                    pen.DeleteObject()
                if hasattr(self, 'hdc'):
                    win32gui.ReleaseDC(self.hwnd, self.hdc)
                self.is_cleaned_up = True
            except Exception as e:
                print(f"Cleanup error: {e}")

    def run(self):
        """Main loop for screen object detection"""
        try:
            print(f"Starting detection on {self.device}... Press 'Q' to quit.")
            
            while True:
                # Capture screen and process
                frame = self.capture_screen()
                current_time = time.time()
                
                # Run detection at specified interval
                if current_time - self.last_detection >= self.detection_interval:
                    results = self.model(frame)
                    detections = results.xyxy[0].cpu().numpy()
                    self.update_tracked_objects(detections)
                    self.last_detection = current_time
                
                # Draw objects and update FPS
                self.draw_objects()
                self.update_fps()
                
                # Check for quit key
                if win32api.GetAsyncKeyState(ord('Q')) & 0x8000:
                    print("Stopping detection...")
                    break
                
                # Small sleep to reduce CPU usage
                time.sleep(0.01)
                
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            self.cleanup()

if __name__ == "__main__":
    try:
        detector = ScreenObjectDetector(confidence=0.5)
        detector.run()
    except Exception as e:
        print(f"Main error: {e}")
