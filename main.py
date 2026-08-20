import os
import cv2
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from vision_processor import VisionProcessor

app = FastAPI(title="Cloud AI Video Batch Processing API")

class VideoAnalysisResult(BaseModel):
    status: str
    total_frames_processed: int
    overall_risk_score: float
    risk_level: str
    environment_warning: Optional[str] = None
    spoof_detected: bool
    spoof_reasons: List[str]
    final_perclos: float
    max_blink_duration_ms: int

def determine_risk_level(score: float) -> str:
    if score >= 80: return "CRITICAL"
    elif score >= 50: return "HIGH"
    elif score >= 20: return "MODERATE"
    return "LOW"

@app.post("/api/analyze-video", response_model=VideoAnalysisResult)
async def analyze_video(file: UploadFile = File(...)):
    """
    Accepts a video file, runs MediaPipe AI across all frames,
    and returns an aggregated risk score.
    """
    if not file.filename.endswith(('.mp4', '.avi', '.mov', '.webm')):
        raise HTTPException(status_code=400, detail="Invalid video format. Must be mp4, avi, mov, or webm.")

    # Save uploaded file to a temporary location for OpenCV to read
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save temp video: {e}")

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        os.remove(tmp_path)
        raise HTTPException(status_code=500, detail="Could not open video file with OpenCV.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps: # Handle nan or 0
        fps = 30.0

    processor = VisionProcessor()
    frames_processed = 0
    final_telemetry = None
    all_spoof_reasons = set()
    max_blink = 0
    bad_lighting_frames = 0

    # Process at max 10 FPS for speed
    target_fps = 10.0
    frame_interval_ms = 1000.0 / target_fps
    next_process_time_ms = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            
            # Skip frames to achieve target FPS
            if current_time_ms < next_process_time_ms:
                continue
                
            next_process_time_ms = current_time_ms + frame_interval_ms
            frames_processed += 1
            timestamp_sec = current_time_ms / 1000.0
            
            # Process the frame
            telemetry = processor.process_frame(frame, timestamp_sec)
            final_telemetry = telemetry
            
            # Aggregate warnings
            if telemetry["spoof_detected"] and telemetry["spoof_reason"]:
                for reason in telemetry["spoof_reason"].split(" | "):
                    all_spoof_reasons.add(reason)
                    
            if telemetry["blink_duration_ms"] > max_blink:
                max_blink = telemetry["blink_duration_ms"]
                
            if telemetry["lighting_condition"] != "NORMAL":
                bad_lighting_frames += 1
                
    finally:
        cap.release()
        os.remove(tmp_path)

    if frames_processed == 0 or not final_telemetry:
        raise HTTPException(status_code=400, detail="Video contained no valid frames.")

    # Calculate overall risk score
    risk_score = final_telemetry["fatigue_score"]
    spoof_detected = len(all_spoof_reasons) > 0
    
    if spoof_detected:
        risk_score += 50.0
    if max_blink > 800:
        risk_score += 20.0
        
    overall_score = min(100.0, risk_score)
    risk_level = determine_risk_level(overall_score)

    environment_warning = None
    if bad_lighting_frames > (frames_processed * 0.5):
        environment_warning = "Warning: More than 50% of the video had poor lighting (HIGH or LOW). Please adjust cabin lighting."

    return VideoAnalysisResult(
        status="success",
        total_frames_processed=frames_processed,
        overall_risk_score=overall_score,
        risk_level=risk_level,
        environment_warning=environment_warning,
        spoof_detected=spoof_detected,
        spoof_reasons=list(all_spoof_reasons),
        final_perclos=final_telemetry["perclos"],
        max_blink_duration_ms=max_blink
    )
