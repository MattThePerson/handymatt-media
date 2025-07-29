import os
import cv2
import numpy as np
import math

from .lib.helpers import get_display_aspect_ratio


#region - PUBLIC -------------------------------------------------------------------------------------------------------

def generateVideoSpritesheet(
        video_path: str,
        output_dir: str,
        filestem: str='spritesheet',
        number_of_frames: int=100,
        height: int=300,
        verbose: bool=False,
    ):
    """ (OpenCV) For a given video, will generate a spritesheet of seek thumbnails (preview thumbnails) as well as .vtt file. """

    if not os.path.exists(video_path):
        raise FileNotFoundError('Video doesnt exist:', video_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")
    
    # Get video properties
    aspect_ratio, frame_count, duration = _getVideoProperties(cap, video_path)
    
    # Extract frames
    thumbnails = _extractFrames(cap, number_of_frames, frame_count)
    cap.release()
    
    # build spritesheet
    thumb_height = height
    thumb_width = int(thumb_height * aspect_ratio)
    spritesheet, vtt_content = _generateSpritesheet(thumbnails, duration, thumb_width, thumb_height, filestem)
    
    # Save spritesheet image
    spritesheet_path = os.path.join(output_dir, f"{filestem}.jpg")
    cv2.imwrite(spritesheet_path, spritesheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    
    # Save VTT file
    vtt_path = os.path.join(output_dir, f"{filestem}.vtt")
    with open(vtt_path, 'w') as f:
        f.write(vtt_content)
    
    return spritesheet_path, vtt_path


#region - PRIVATE ------------------------------------------------------------------------------------------------------


def _getVideoProperties(cap: cv2.VideoCapture, video_path: str):
    aspect_ratio = get_display_aspect_ratio(video_path)
    if aspect_ratio is None:
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        aspect_ratio = video_width / video_height
    # print('aspect_ratio:', aspect_ratio)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps
    
    return aspect_ratio, frame_count, duration



def _extractFrames(cap: cv2.VideoCapture, frames_to_extract: int, total_frames: int):
    step = max(30, total_frames / frames_to_extract)
    frames = []
    for i in range(frames_to_extract):
        frame_pos = int((i+0.5) * step)
        print('\rextracting frame {}/{} [{}/{}]'.format(step, i+1, frames_to_extract, frame_pos, total_frames), end='')
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    print()
    return frames


def _generateSpritesheet(thumbnails: list, duration: float, thumb_width: int, thumb_height: int, filestem: str):
    
    # Determine optimal grid layout for spritesheet (aim for roughly square)
    thumb_count = len(thumbnails)
    cols = int(math.ceil(math.sqrt(thumb_count)))
    rows = int(math.ceil(thumb_count / cols))
    
    # Create blank spritesheet image
    spritesheet_width = cols * thumb_width
    spritesheet_height = rows * thumb_height
    spritesheet = np.zeros((spritesheet_height, spritesheet_width, 3), dtype=np.uint8)
    
    # Prepare VTT file content
    vtt_content = "WEBVTT\n\n"
    
    for i, frame in enumerate(thumbnails):
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
        start_time = i * (duration / thumb_count)
        end_time = (i + 1) * (duration / thumb_count)
        
        # Format times as HH:MM:SS.mmm
        start_time_str = _format_time(start_time)
        end_time_str = _format_time(end_time)
        
        # Add entry to VTT file
        vtt_content += f"{start_time_str} --> {end_time_str}\n"
        vtt_content += f"{filestem}.jpg#xywh={x},{y},{thumb_width},{thumb_height}\n\n"

    return spritesheet, vtt_content


def _format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format for VTT files."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

