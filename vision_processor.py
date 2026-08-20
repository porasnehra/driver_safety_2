import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import cv2
import mediapipe as mp
import numpy as np
import time
from typing import Dict, Tuple, List

# MediaPipe Initialization
mp_face_mesh = mp.solutions.face_mesh

RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_CORNERS = [61, 291]

class VisionProcessor:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        self.reset_state()

    def reset_state(self):
        self.eye_closed_frames = 0
        self.total_frames = 0
        self.blink_start_time = 0
        self.is_blinking = False
        self.last_blink_duration = 0
        self.spoof_flags = []
        self.ear_history = []
        self.mouth_history = []
        
    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = self.clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final

    def _evaluate_lighting(self, frame: np.ndarray) -> str:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        if avg_brightness < 40: return "LOW_LIGHT"
        elif avg_brightness > 210: return "HIGH_LIGHT"
        else: return "NORMAL"

    def _euclidean_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _calculate_ear(self, landmarks, eye_indices) -> float:
        p1 = landmarks[eye_indices[0]]
        p4 = landmarks[eye_indices[3]]
        p2 = landmarks[eye_indices[1]]
        p6 = landmarks[eye_indices[5]]
        p3 = landmarks[eye_indices[2]]
        p5 = landmarks[eye_indices[4]]
        width = self._euclidean_distance(p1, p4)
        height1 = self._euclidean_distance(p2, p6)
        height2 = self._euclidean_distance(p3, p5)
        if width == 0: return 0.0
        return (height1 + height2) / (2.0 * width)

    def _check_duchenne_marker(self, landmarks, left_ear, right_ear) -> bool:
        mouth_left = landmarks[MOUTH_CORNERS[0]]
        mouth_right = landmarks[MOUTH_CORNERS[1]]
        mouth_width = self._euclidean_distance(mouth_left, mouth_right)
        face_width = self._euclidean_distance(landmarks[LEFT_EYE[3]], landmarks[RIGHT_EYE[0]])
        normalized_mouth_width = mouth_width / face_width if face_width > 0 else 0
        
        is_smiling = normalized_mouth_width > 0.55  # Increased threshold to reduce false positives
        avg_ear = (left_ear + right_ear) / 2.0
        eyes_wide = avg_ear > 0.35  # Increased threshold

        return is_smiling and eyes_wide

    def process_frame(self, frame: np.ndarray, timestamp_sec: float) -> Dict:
        self.total_frames += 1
        enhanced_frame = self._apply_clahe(frame)
        rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        lighting_status = self._evaluate_lighting(frame)
        
        telemetry = {
            "face_detected": False,
            "lighting_condition": lighting_status,
            "ear": 0.0,
            "perclos": 0.0,
            "blink_duration_ms": 0,
            "fatigue_score": 0.0,
            "spoof_detected": False,
            "spoof_reason": ""
        }

        if results.multi_face_landmarks:
            telemetry["face_detected"] = True
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = [(lm.x, lm.y) for lm in face_landmarks.landmark]
            
            left_ear = self._calculate_ear(landmarks, LEFT_EYE)
            right_ear = self._calculate_ear(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0
            telemetry["ear"] = round(avg_ear, 3)
            self.ear_history.append(avg_ear)
            
            mouth_left = landmarks[MOUTH_CORNERS[0]]
            mouth_right = landmarks[MOUTH_CORNERS[1]]
            mouth_width = self._euclidean_distance(mouth_left, mouth_right)
            self.mouth_history.append(mouth_width)

            EAR_THRESHOLD = 0.21
            if avg_ear < EAR_THRESHOLD:
                self.eye_closed_frames += 1
                if not self.is_blinking:
                    self.is_blinking = True
                    self.blink_start_time = timestamp_sec
            else:
                if self.is_blinking:
                    self.is_blinking = False
                    blink_end_time = timestamp_sec
                    duration_ms = int((blink_end_time - self.blink_start_time) * 1000)
                    if duration_ms > 0:
                        self.last_blink_duration = duration_ms
            
            telemetry["blink_duration_ms"] = self.last_blink_duration
            perclos = (self.eye_closed_frames / self.total_frames) * 100
            telemetry["perclos"] = round(perclos, 2)

            fatigue = 0
            if perclos > 15.0: fatigue += 20
            if perclos > 25.0: fatigue += 30
            if self.last_blink_duration > 500: fatigue += 20
            if self.last_blink_duration > 1000: fatigue += 30
            telemetry["fatigue_score"] = min(fatigue, 100)
            
            if len(self.spoof_flags) > 0:
                telemetry["spoof_detected"] = True
                telemetry["spoof_reason"] = " | ".join(list(set(self.spoof_flags)))
                self.spoof_flags = []
                
        return telemetry

    def finalize_spoof_check(self) -> List[str]:
        """
        Runs at the end of the video batch.
        Calculates the standard deviation of facial movements.
        If a face is perfectly static (e.g. a printed photograph),
        the variance will be extremely close to 0.
        """
        final_flags = []
        if len(self.ear_history) > 10 and len(self.mouth_history) > 10:
            ear_variance = np.std(self.ear_history)
            mouth_variance = np.std(self.mouth_history)
            
            # A real human face always has micro-movements (> 0.001)
            # A printed photo will have variance < 0.0005 depending on camera noise
            if ear_variance < 0.001 and mouth_variance < 0.001:
                final_flags.append("Static Image Detected (Zero Micro-Movements)")
                
        return final_flags
