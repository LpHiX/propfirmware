import os
import subprocess
import datetime
import sys

# --- Configuration ---
RTSP_URL = "rtsp://[2a0c:5bc0:40:2e26:e821:aff:fe50:2bb9]:8080/h264_opus.sdp"
CAPTURE_FPS = "1"   # Captures 1 frame per second
PLAYBACK_FPS = "30" # Plays back at 30 frames per second
# ---------------------

def main():
    # 1. Generate current date and time strings
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d") # e.g., 2026-04-23
    time_str = now.strftime("%H-%M-%S") # e.g., 14-20-43

    # 2. Set up the dynamic paths
    base_dir = rf"D:\Videos\FYPTesting\{date_str}"
    frames_dir = os.path.join(base_dir, "timelapse_frames", time_str)
    
    # Final video name: timelapse_YYYY-MM-DD HH-MM-SS.mp4
    output_video_name = f"timelapse_{date_str} {time_str}.mp4"
    output_video_path = os.path.join(base_dir, output_video_name)

    # 3. Create the directories (exist_ok=True prevents crashes if the folder is already there)
    print(f"📁 Setting up directories in: {base_dir}")
    os.makedirs(frames_dir, exist_ok=True)

    # 4. Build the capture command
    frames_pattern = os.path.join(frames_dir, "frame_%06d.jpg")
    capture_cmd = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-i", RTSP_URL,
        "-r", CAPTURE_FPS,
        "-f", "image2",
        frames_pattern
    ]

    print("\n" + "="*60)
    print(f"🎥 STARTING CAPTURE (1 frame per {CAPTURE_FPS} sec)")
    print(f"📂 Saving frames to: {frames_dir}")
    print("⏳ Let this run for as long as you want.")
    print("🛑 PRESS [CTRL + C] TO STOP AND GENERATE VIDEO")
    print("="*60 + "\n")

    # 5. Run capture and wait for user to hit Ctrl+C
    try:
        subprocess.run(capture_cmd)
    except KeyboardInterrupt:
        pass

    print("\n" + "="*60)
    print("⚙️ CAPTURE STOPPED. STITCHING VIDEO TOGETHER...")
    print("="*60 + "\n")

    # 6. Build the stitching command
    stitch_cmd = [
        "ffmpeg",
        "-y", 
        "-framerate", PLAYBACK_FPS,
        "-i", frames_pattern,
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        output_video_path
    ]

    # 7. Run the stitch command
    subprocess.run(stitch_cmd)

    print("\n" + "="*60)
    print(f"✅ DONE! Your timelapse is ready:")
    print(f"🎞️  {output_video_path}")
    print(f"🗑️  If it looks good, you can delete the '{frames_dir}' folder to save space.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()