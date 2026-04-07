import os
import time
import shutil

# Configuration
WATCH_FILE = 'checkpoints/results/inpaint/joint/000001.png'
HISTORY_DIR = 'checkpoints/results/history_FULLDATASET'
CHANGE_THRESHOLD = 1  # Copy every 10 changes

def monitor():
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)
        print(f"Created history directory: {HISTORY_DIR}")

    print(f"Monitoring {WATCH_FILE}...")
    print(f"Copies will be saved to {HISTORY_DIR} every {CHANGE_THRESHOLD} updates.")

    last_mtime = 0
    change_count = 0
    version = 1

    while True:
        if os.path.exists(WATCH_FILE):
            current_mtime = os.path.getmtime(WATCH_FILE)
            
            if current_mtime != last_mtime:
                last_mtime = current_mtime
                change_count += 1
                
                print(f"Update detected ({change_count}/{CHANGE_THRESHOLD})")
                
                if change_count >= CHANGE_THRESHOLD:

                    dest_file = os.path.join(HISTORY_DIR, f"000001_v{version:03d}.png")
                    shutil.copy2(WATCH_FILE, dest_file)
                    print(f" >>> Saved version {version} to {dest_file}")
                    version += 1
                    change_count = 0
        else:
            # Wait for the file to be created if it doesn't exist yet
            pass

        time.sleep(5) # Check every 5 seconds

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
