import cv2
import sys
from pathlib import Path

def parse_time(time_str):
    """Parses mm:ss or seconds string into total seconds."""
    if ':' in time_str:
        m, s = map(int, time_str.split(':'))
        return m * 60 + s
    return int(time_str)

def extract_reference():
    # Resolve path relative to this script
    current_dir = Path(__file__).parent
    video_path = current_dir / "input_video.mp4"
    
    if not video_path.exists():
        print(f"Error: {video_path} not found.")
        return

    print(f"Video: {video_path.name}")
    
    # Interactive Inputs
    try:
        time_input = input("Enter timestamp (MM:SS or seconds) for reference frame: ")
        timestamp = parse_time(time_input)
        
        cam_name = input("Enter camera name (e.g., 'main', 'cam2', 'wide'): ").strip()
        if not cam_name:
            cam_name = "main" # default
            
        filename = f"reference_{cam_name}.jpg"
        output_path = current_dir / filename
        
    except ValueError:
        print("Invalid format. Use MM:SS or integer seconds.")
        return

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame = int(timestamp * fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if target_frame >= total_frames:
        print(f"Error: Timestamp {timestamp}s is beyond video duration.")
        cap.release()
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    success, frame = cap.read()

    if success:
        cv2.imwrite(str(output_path), frame)
        print(f"\n✅ Saved {filename}")
        print(f"   Timestamp: {timestamp}s (Frame {target_frame})") 
        print(f"   Camera ID: {cam_name}")
    else:
        print("Error: Could not read frame from video.")

    cap.release()

if __name__ == "__main__":
    extract_reference()
