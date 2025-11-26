import os, logging, shutil, subprocess, time, sys

def is_video_corrupted(video_path):
    """Return True if video is corrupted (FFmpeg detects errors)."""
    cmd = ["ffmpeg", "-v", "error", "-i", video_path, "-f", "null", "-"]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    return len(result.stderr) > 0


def validate_videos_in_folder(folder_path, move_corrupted=True):
    corrupted_dir = os.path.join(folder_path, "corrupted")
    os.makedirs(corrupted_dir, exist_ok=True)

    mp4_files = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]

    logging.info(f"🔍 Checking {len(mp4_files)} videos in '{folder_path}'...\n")

    for filename in mp4_files:
        video_path = os.path.join(folder_path, filename)

        if is_video_corrupted(video_path):
            logging.error(f"❌ Corrupted: {filename}")
            if move_corrupted:
                shutil.move(video_path, os.path.join(corrupted_dir, filename))
        else:
            logging.info(f"✅ OK: {filename}")

    logging.info("\n✅ Validation complete!")


if __name__ == '__main__':
    folder = os.getcwd() + '/videos_full/'

    logging.basicConfig(filename='validate_{}.log'.format(int(time.time())), filemode='w', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info('Start validating videos.')

    validate_videos_in_folder(folder, move_corrupted=False)