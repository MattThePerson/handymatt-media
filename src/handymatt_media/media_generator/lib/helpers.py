import subprocess

def get_display_aspect_ratio(video_path) -> float|None:
    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=display_aspect_ratio',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        dar = result.stdout.strip()
        if ':' in dar:
            w, h = map(int, dar.split(':'))
            return w / h
        else:
            return float(dar)  # fallback if already a float
    except Exception as e:
        print(f"Error getting DAR: {e}")
        return None
