import os
import subprocess
import shlex
from pathlib import Path
import time
import shutil
import math
import numpy as np

import cv2
from nudenet import NudeDetector

from .lib.stills import extract_stills_from_video, addDetectionsToImage, get_detections_score, floodingMethod


# GENERATE VIDEO TEASER
def generateVideoTeaser(input_path, output_dir, savename, abs_amount_mode=False, n=10, jump=300, clip_len=1.3, start_perc=5, end_perc=95, keep_clips=False, skip=1, smallSize=False, quit=True):
    if not os.path.exists(input_path):
        print("ERROR: Path doesn't exist [{}]".format(input_path))
        return ""
    savepath = os.path.join( output_dir, savename )
    smallRes = "640:360"
    print("Generating preview for video:", input_path)
    import subprocess
    duration_command = "ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1"
    duration_sec = int(float(subprocess.run(duration_command.split(" ") + [input_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout))
    times = []
    start_t = duration_sec * start_perc / 100
    end_t = duration_sec * end_perc / 100
    if abs_amount_mode:
        jump = (end_t - start_t) / n
    t = start_t
    skipCount = skip
    while t < end_t:
        skipCount -= 1
        if skipCount == 0:
            times.append(_formatSeconds(t))
            skipCount = skip
        t += jump
    tempnames = []
    for i, time in enumerate(times):
        print("\rGenerating clip ({}/{}) at time [{}]".format(i+1, len(times), time), end='')
        tempname = os.path.join( output_dir, f'temp_{i+1}.mp4' )
        command = [
            'ffmpeg', '-ss', time,
            '-i', input_path,
            '-t', f'00:00:{clip_len}',
            '-map', '0:v:0', '-an',
            '-c:v', 'libx264',
            '-v', 'error'
            #, '-stats',
        ]
        if smallSize:
            command.extend([
                '-vf', f'scale={smallRes}'
            ])
        command.extend([
            tempname, '-y',
        ])
        # print(' '.join(command))
        subprocess.run(command)
        tempnames.append(tempname)
    print()
    print("Concatenating {} clips ...".format(len(tempnames)))
    savepath = _concatClips(savepath, tempnames)
    if not keep_clips:
        for clip in tempnames:
            os.remove(clip)
    print("Done.")
    return savepath



def _concatClips(savepath: str, clips: list[str]) -> str:
    command = [ 'ffmpeg' ]
    filter_command = ''
    for i, clip in enumerate(clips):
        command.extend(["-i", clip]) # add clips to command
        filter_command += f" [{i}:v]"
    filter_command += f'concat=n={len(clips)}:v=1 [v]'
    command.extend([
        '-filter_complex', filter_command, # filter
        '-map', '[v]',
        '-v', 'error', '-stats',
        savepath, '-y',
    ])
    # print('running command [{}]'.format(command))
    _ = subprocess.run(command)
    return savepath


# CREATE GIF
def create_gif(videopath, savepath, start_time_sec, gif_duration=7, resolution=720, fps=15):
    print("Creating gif for video at path: [{}]".format(videopath))
    savedir = Path(savepath).parent
    temppath = os.path.join(savedir, "temp.gif")
    if not savepath.endswith('.mp4'):
        savepath = savepath + '.mp4'
    create_gif_command = f'ffmpeg -i "{videopath}" -ss {int(start_time_sec)} -t {gif_duration} -vf "fps={fps},scale=-1:{resolution}:flags=lanczos" -c:v gif "{temppath}" -y'
    os.system(create_gif_command)
    convert_to_mp4_command = f'ffmpeg -i "{temppath}" -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" "{savepath}"'
    print("Converting gif to mp4")
    os.system(convert_to_mp4_command)
    print("Deleting temp gif")
    os.remove(temppath)
    if not os.path.exists(savepath):
        return None
    return savepath



#### HELPER FUNCTIONS ####

def _formatSeconds(sec):
    h = int(sec / 3600)
    sec -= h * 3600
    m = int(sec / 60)
    sec -= m*60
    s = int(sec)
    return f"{h}:{m}:{s}"

# STUPID! Should be replaced with shlex
def _getCommandTerms(command):
    terms = []
    quotesOpen = False
    term = []
    for c in list(command):
        if not quotesOpen:
            if c == " ":
                if len(term) > 0: terms.append("".join(term))
                term = []
            elif c == '"':
                quotesOpen = True
            else:
                term.append(c)
        else:
            if c == '"':
                quotesOpen = False
                if len(term) > 0: terms.append("".join(term))
                term = []
            else:
                term.append(c)
    if len(term) > 0: terms.append("".join(term))
    return terms



#### PREVEIW THUMBS #####

# resolution can be a list of resolutions
def extractPreviewThumbs(video_path: str, target_dir: str, amount=5, resolution:list[int]|int=720, n_frames=30*10, keep_temp_stills=False, show_detections=False) -> list[str]:
    start = time.time()
    if not isinstance(resolution, list):
        resolution = [resolution]
    if not os.path.exists(video_path):
        print('Video path doesnt exist:', video_path)
        exit()
    temp_folder = os.path.join( target_dir, 'temp' )
    os.makedirs(temp_folder, exist_ok=True)
    temp_folder_contents = os.listdir(temp_folder)
    if temp_folder_contents != []:
        print('Loaded {} existing temp stills from dir: {}'.format(len(temp_folder_contents), temp_folder))
        stills = [ (os.path.join(temp_folder, f) ,) for f in temp_folder_contents ]
    else:
        print('Generating stills ...')
        stills = extract_stills_from_video(video_path, temp_folder, fn_root='temp', jump_frames=n_frames, start_perc=2, end_perc=40, top_stillness=60)

    # Convert to dict and load cv img
    image_items = []
    for i in range(len(stills)):
        item = stills[i]
        obj = { key: val for key, val in zip(['path', 'stillness', 'sharpness'], item) }
        image_items.append(obj)
    image_items.sort(key=lambda x: x['path'])

    # Analyse stills
    nd = NudeDetector()
    score = None
    for obj in image_items:
        img_path = obj['path']
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # print(img_path)
        detections = nd.detect(img_path)
        obj['detections'] = detections
        if show_detections:
            addDetectionsToImage(image, detections)
            cv2.putText(obj['image'], f'score: {score}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 220, 100), 2, cv2.LINE_AA)
        score = get_detections_score(detections, image.shape)
        obj['score'] = score
        obj['image'] = image
    image_items.sort(reverse=True, key=lambda obj: obj['score'])
    
    image_items_flood = floodingMethod(image_items, stills_amount=5)

    # Save images
    image_paths = []
    for res in resolution:
        for i, item in enumerate(image_items_flood, start=1):
            savepath = os.path.join( target_dir, 'previewThumb_{}_{}_[{}].png'.format(res, i, int(item['score']*100)) )
            image_paths.append(savepath)
            ar = item['image'].shape[1] / item['image'].shape[0]
            img = cv2.resize(item['image'], (int(res*ar), res))
            cv2.imwrite(savepath, img)
    
    if not keep_temp_stills:
        shutil.rmtree(temp_folder)
    
    print('Done. Took {:.4f}s'.format((time.time()-start)))
    return image_paths


# SEEK THUMBNAILS


# 
def generateSeekThumbnails(video_path: str, output_dir: str, filename: str='seekthumbs', n: int=100, height: int=300):
    """ For a given video, will generate a spritesheet of seek thumbnails (preview thumbnails) as well as .vtt file. """
    if not os.path.exists(video_path):
        raise FileNotFoundError('Video doesnt exist:', video_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = video_width / video_height
    
    # Calculate thumbnail dimensions maintaining aspect ratio
    thumb_height = height
    thumb_width = int(thumb_height * aspect_ratio)
    
    # Calculate the step between frames to get n evenly spaced frames
    step = max(1, frame_count // n)
    
    # Determine optimal grid layout for spritesheet (aim for roughly square)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    
    # Create blank spritesheet image
    spritesheet_width = cols * thumb_width
    spritesheet_height = rows * thumb_height
    spritesheet = np.zeros((spritesheet_height, spritesheet_width, 3), dtype=np.uint8)
    
    # Prepare VTT file content
    vtt_content = "WEBVTT\n\n"
    
    # Extract frames and build spritesheet
    for i in range(n):
        print('\rextracting frame {}/{}'.format(i+1, n), end='')
        # Calculate frame position and timestamp
        frame_pos = min(i * step, frame_count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame to thumbnail size
        thumbnail = cv2.resize(frame, (thumb_width, thumb_height))
        
        # Calculate position in spritesheet
        row = i // cols
        col = i % cols
        x = col * thumb_width
        y = row * thumb_height
        
        # Paste thumbnail into spritesheet
        spritesheet[y:y+thumb_height, x:x+thumb_width] = thumbnail
        
        # Calculate timestamps for VTT
        start_time = i * (duration / n)
        end_time = (i + 1) * (duration / n)
        
        # Format times as HH:MM:SS.mmm
        start_time_str = _format_time(start_time)
        end_time_str = _format_time(end_time)
        
        # Add entry to VTT file
        vtt_content += f"{start_time_str} --> {end_time_str}\n"
        vtt_content += f"{filename}.jpg#xywh={x},{y},{thumb_width},{thumb_height}\n\n"
    print()
    
    # Save spritesheet image
    spritesheet_path = os.path.join(output_dir, f"{filename}.jpg")
    cv2.imwrite(spritesheet_path, spritesheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    
    # Save VTT file
    vtt_path = os.path.join(output_dir, f"{filename}.vtt")
    with open(vtt_path, 'w') as f:
        f.write(vtt_content)
    
    # Release video capture
    cap.release()
    
    return spritesheet_path, vtt_path


def _format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format for VTT files."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

