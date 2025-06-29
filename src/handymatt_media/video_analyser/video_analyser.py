from typing import Any
import cv2
import imagehash
import hashlib
from PIL import Image
import subprocess
import subprocess
import datetime
import os
# import numpy as np



# v2
# Updated (before) 19.03.2024
def getVideoHash_Old(video_path):
    import os
    import subprocess
    filesize = os.path.getsize(video_path)
    if video_path.endswith("mkv") or video_path.endswith("webm"):
        duration_command = "ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1"
    else:
        duration_command = "ffprobe -v quiet -select_streams v:0 -show_entries stream=duration -of default=noprint_wrappers=1:nokey=1"
    try:
        duration = float(subprocess.run(duration_command.split(" ") + [video_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout)
    except:
        return -1
    width = int(subprocess.run("ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=s=x:p=0".split(" ") + [video_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout)
    height = int(subprocess.run("ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=s=x:p=0".split(" ") + [video_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout)
    string = str(filesize) + str(duration) + str(width) + str(height)
    return _myHashFunction(string)


## NEW HASH USING FFMPEF


def getVideoHash_openCV(video_path: str, start_perc: float=0.2, num_frames: int=5) -> str:
    """ Get video hash using open cv """
    cap = cv2.VideoCapture(video_path)
    bytes_list = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start_perc * frame_count)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        bytes_list.append(frame.tobytes())
    cap.release()
    if not bytes_list:
        raise Exception('No bytes read from video:', video_path)
    combined = b"".join(bytes_list)
    sha256_hash = hashlib.sha256(combined)
    return sha256_hash.hexdigest()[:12] # convert to hex and return first 12 digits


# def getVideoHash_ffmpegPython(video_path: str, timestamp: float=10.0, duration: float=20.0) -> str:
#     """ Get video hash using ffmpeg-python to extract raw frame bytes. """
#     import ffmpeg
#     out, _ = (
#         ffmpeg
#         .input(video_path, ss=timestamp, t=duration)
#         .output('pipe:', format='rawvideo', pix_fmt='rgb24')
#         .run(capture_stdout=True, capture_stderr=True)
#     )
#     sha256_hash = hashlib.sha256(out)
#     return sha256_hash.hexdigest()[:12] # convert to hex and return first 12 digits


def getVideoHash_ffmpeg(video_path: str) -> str:
    """ Get video hash based on byte chunks etracted from video stream using ffmpeg. Outputted hash is 12 digit hex string """
    video_data = extract_video_bytes(str(video_path), [10.0], 20)
    sha256_hash = hashlib.sha256(video_data)
    return sha256_hash.hexdigest()[:12] # convert to hex and return first 12 digits


def extract_video_bytes(video_path: str, timestamps, duration: float=20):
    """ Extract raw bytes of given duration at timestams from a video """
    extracted_bytes = []
    for ts in timestamps:
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-c:v", "copy",    # No transcoding, copy raw video stream
            "-map", "0:v:0",   # Extract only the video stream
            "-ss", str(ts),
            "-t", str(duration),
            "-f", "rawvideo",  # Output raw video data
            "pipe:1"           # Output to stdout
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        extracted_bytes.append(result.stdout)
    return b"".join(extracted_bytes)

## END HASH


# v1
# NOTE:
# - from testing, changing params to start_percs=[0] & num_frames=14 hash time went from  ~712ms -> ~250ms
def getVideoHash(filepath, start_percs=[0.05, 0.5, 0.85], num_frames=3, quiet=True, show_string=False) -> str|None:
# def getVideoHash_OpenCV(filepath, start_percs=[0.2], num_frames=10, quiet=True, show_string=False):
    if not quiet: print('opening')
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise ValueError("Cannot open video file.")
    
    if not quiet: print('reading')
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count_snipped = (frame_count-frame_count%33)
    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_hashes = []

    for start_perc in start_percs:
        if not quiet: print('setting')
        start_frame = int(frame_count_snipped * start_perc)
        succ = cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        if not succ:
            print('\nUnable to set cap to {} of video for:\n  "{}"'.format(start_perc, filepath))
            cap.release()
            return None

        # Read 'num_frames' sequentially
        for _ in range(num_frames):
            if not quiet: print('  reading frame ...')
            ret, frame = cap.read()
            if not ret:
                print('\nUnable to read frame {}+x for "{}"'.format(start_frame, filepath))
                cap.release()
                return None
            if not quiet: print('    hashing')
            frame_hashes.append(hash_frame(frame))
    
    cap.release()

    string = ''
    for param in [frame_count_snipped, frame_width, frame_height]:
        string += str(param) + ' '
    string += ' '.join(frame_hashes)
    if show_string: print(string)
    return _myHashFunction_12digits(string)


def hash_frame(frame):
    size = 16
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((size, size), Image.Resampling.LANCZOS)
    imghsh = str(imagehash.average_hash(pil_image)) # Using average hash for simplicity
    return imghsh
    return _myHashFunction_12digits(imghsh)



# v1.2
def _myHashFunction_12digits(string):
    string = str(string)
    hex_digits = 12 # number of hex digits in hash
    bitwidth = 4*hex_digits
    mask = ((1 << bitwidth) - 1)
    x = 100
    for i, c in enumerate(list(string)):
        n = ord(c)
        x = (n * x + 2 ** (i+10)) & mask
        x = x ^ (x >> 16)
    a, b, c, d = 51, 9323, 83, 573438
    x = (a * x + b) & mask
    x = x ^ (x >> 16)
    x = (c * x + d) & mask
    x = x ^ (x >> 16)
    hexstr = str(hex(x))[2:]
    return "0"*(hex_digits-len(hexstr)) + hexstr

# v1
def _myHashFunction(string):
    hex_digits = 8 # number of hex digits in hash
    bitwidth = 4*hex_digits
    mask = ((1 << bitwidth) - 1)
    x = 100
    for i, c in enumerate(list(string)):
        n = ord(c)
        x = (n * x + 2 ** (i+10)) & mask
        x = x ^ (x >> 16)
    a, b, c, d = 51, 9323, 83, 573438
    x = (a * x + b) & mask
    x = x ^ (x >> 16)
    x = (c * x + d) & mask
    x = x ^ (x >> 16)
    hexstr = str(hex(x))[2:]
    return "0"*(hex_digits-len(hexstr)) + hexstr


# 
def getVideoData(filepath: str) -> dict[str, Any]:
    duration_command = "ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1"
    height_command = "ffprobe -v error -select_streams v:0 -show_entries stream=height -of default=nw=1:nk=1"
    fps_command = "ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate"
    duration_sec = int(float(subprocess.run(duration_command.split(" ") + [filepath], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout))
    duration_sec = max(1, duration_sec)
    duration = str(datetime.timedelta(seconds=duration_sec))
    filesize_mb = round(os.stat(filepath).st_size / (1024 * 1024), 3)
    bitrate = int(filesize_mb * 8 * 1024 / duration_sec)
    height = int(float(subprocess.run(height_command.split(" ") + [filepath], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout))
    fps_stdout = subprocess.run(fps_command.split(" ") + [filepath], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
    fps_int = _get_fps_from_ffprobe_fps_stdout(fps_stdout)
    return {
        "duration" : duration,
        "duration_seconds" : duration_sec,
        "bitrate" : bitrate,
        "resolution" : height,
        "filesize_mb" : filesize_mb,
        "fps" : fps_int
    }

def _get_fps_from_ffprobe_fps_stdout(_input) -> int:
    """ converts fps stdout from format: `b'30/1\\r\\n'` to integer: int(30/1) """
    string = str(_input) # convert to str
    string = string.replace(r"\r\n'", '') # replace windows EOL + `'`
    string = string.replace(r"\n'", '') # replace linux EOL + `'`
    parts = string[2:].split("/") # 
    if len(parts) == 2:
        return int(round(float(parts[0]) / float(parts[1]), 0))
    else:
        return int(parts[0])


# Command Line Interface
def cli():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-hash', help='Return video hash')
    args = parser.parse_args()
    
    if args.hash:
        video = args.hash
        hash = getVideoHash(video)
        print("[{}] is the hash for '{}'".format(hash, video))

    print()
