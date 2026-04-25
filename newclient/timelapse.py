import os
import subprocess
import datetime
import time

# --- Configuration ---
RTSP_URL = "rtsp://[2a0c:5bc0:40:2e26:e821:aff:fe50:2bb9]:8080/h264_opus.sdp"
CAPTURE_FPS = "1"   
PLAYBACK_FPS = "30" 
WATCHDOG_TIMEOUT = 20 # Seconds to wait without a new frame before restarting
# ---------------------

def get_frame_count(directory):
    """Quickly counts how many .jpg files are in the folder."""
    try:
        return len([f for f in os.listdir(directory) if f.endswith(".jpg")])
    except FileNotFoundError:
        return 0

def main():
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")

    base_dir = rf"D:\Videos\FYPTesting\{date_str}"
    frames_dir = os.path.join(base_dir, "timelapse_frames", time_str)
    
    output_video_name = f"timelapse_{date_str} {time_str}.mp4"
    output_video_path = os.path.join(base_dir, output_video_name)

    os.makedirs(frames_dir, exist_ok=True)
    frames_pattern = os.path.join(frames_dir, "frame_%06d.jpg")

    print("\n" + "="*60)
    print(f"🐶 STARTING WATCHDOG CAPTURE (1 frame per {CAPTURE_FPS} sec)")
    print(f"📂 Saving frames to: {frames_dir}")
    print("🛑 PRESS [CTRL + C] TO STOP AND GENERATE VIDEO")
    print("="*60 + "\n")

    keep_capturing = True
    
    while keep_capturing:
        start_number = get_frame_count(frames_dir) + 1
        
        capture_cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-i", RTSP_URL,
            "-r", CAPTURE_FPS,
            "-start_number", str(start_number),
            "-f", "image2",
            frames_pattern
        ]

        print(f"▶️ [SYSTEM] Launching FFmpeg... (Resuming at: frame_{start_number:06d}.jpg)")
        
        # Use Popen to run FFmpeg in the background silently
        process = subprocess.Popen(capture_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        last_frame_count = get_frame_count(frames_dir)
        seconds_without_new_frame = 0

        try:
            # The Watchdog Loop
            while process.poll() is None: 
                time.sleep(2) 
                
                current_frame_count = get_frame_count(frames_dir)
                
                if current_frame_count > last_frame_count:
                    # Everything is working! Reset the timer.
                    seconds_without_new_frame = 0
                    last_frame_count = current_frame_count
                else:
                    # No new frame arrived. Increase the warning timer.
                    seconds_without_new_frame += 2
                    
                if seconds_without_new_frame >= WATCHDOG_TIMEOUT:
                    print(f"\n💀 [WARNING] ZOMBIE DETECTED! No new frames for {WATCHDOG_TIMEOUT}s.")
                    print("🔌 [SYSTEM] Killing frozen connection and restarting...")
                    process.terminate() 
                    time.sleep(2) 
                    break # Break watchdog loop to restart FFmpeg

        except KeyboardInterrupt:
            print("\n🛑 [SYSTEM] Keyboard interrupt detected. Shutting down safely...")
            process.terminate()
            keep_capturing = False

    print("\n" + "="*60)
    print("⚙️ [SYSTEM] CAPTURE STOPPED. STITCHING VIDEO TOGETHER...")
    print("="*60 + "\n")

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

    # Let the stitching process output text so you can see the progress bar
    subprocess.run(stitch_cmd)

    print("\n" + "="*60)
    print(f"✅ DONE! Your timelapse is ready:")
    print(f"🎞️  {output_video_path}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()