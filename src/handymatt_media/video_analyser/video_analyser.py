import cv2
import imagehash
from PIL import Image
import numpy as np


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


# v1
def getVideoHash(filepath, start_percs=[0.05, 0.5, 0.85], num_frames=3, quiet=True, show_string=False):
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
            return -1

        # Read 'num_frames' sequentially
        for _ in range(num_frames):
            if not quiet: print('  reading frame ...')
            ret, frame = cap.read()
            if not ret:
                print('\nUnable to read frame for')
                cap.release()
                return -1
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
def getVideoData(filepath):
    import subprocess
    import datetime
    import os
    duration_command = "ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1"
    height_command = "ffprobe -v error -select_streams v:0 -show_entries stream=height -of default=nw=1:nk=1"
    fps_command = "ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate"
    duration_sec = int(float(subprocess.run(duration_command.split(" ") + [filepath], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout))
    duration_sec = max(1, duration_sec)
    duration = str(datetime.timedelta(seconds=duration_sec))
    filesize_mb = round(os.stat(filepath).st_size / (1024 * 1024), 3)
    bitrate = int(filesize_mb * 8 * 1024 / duration_sec)
    height = int(float(subprocess.run(height_command.split(" ") + [filepath], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout))
    fps = str(subprocess.run(fps_command.split(" ") + [filepath], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout)
    fps = fps[2:-5]
    parts = fps.split("/")
    if len(parts) == 2:
        fps_int = int(round(float(parts[0]) / float(parts[1]), 0))
    else:
        fps_int = int(parts[0])
    return {
        "duration" : duration,
        "duration_seconds" : duration_sec,
        "bitrate" : bitrate,
        "resolution" : height,
        "filesize_mb" : filesize_mb,
        "fps" : fps_int
    }



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
