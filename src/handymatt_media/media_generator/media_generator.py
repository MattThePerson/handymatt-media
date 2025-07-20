import os
import time
import shutil
import math
import numpy as np

import cv2

from .lib.stills import extract_stills_from_video, addDetectionsToImage, get_detections_score, floodingMethod



#region - PREVIEW THUMBS -----------------------------------------------------------------------------------------------

# resolution can be a list of resolutions
def extractPreviewThumbs(
        video_path: str,
        target_dir: str,
        amount=5,
        resolution:list[int]|int=720,
        n_frames=30*10,
        keep_temp_stills=False,
        show_detections=False
    ) -> list[str]:
    """  """
    from nudenet import NudeDetector

    start = time.time()
    if not isinstance(resolution, list):
        resolution = [resolution]
    if not os.path.exists(video_path):
        raise FileNotFoundError('Video path doesnt exist:', video_path)
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
    
    image_items_flood = floodingMethod(image_items, stills_amount=amount)

    # delete previous preview thumbs (dont delete temp files)
    from send2trash import send2trash
    for filename in os.listdir(target_dir):
        filepath = os.path.normpath( os.path.join(target_dir, filename) )
        if os.path.isfile(filepath):
            send2trash(filepath)

    # Save images
    image_paths = []
    for res in resolution:
        for i, item in enumerate(image_items_flood, start=1):
            savepath = os.path.join( target_dir, 'previewThumb_{}_{}_[{}].png'.format(res, i, int(item['score']*100)) )
            # print('saving:', savepath)
            image_paths.append(savepath)
            ar = item['image'].shape[1] / item['image'].shape[0]
            img = cv2.resize(item['image'], (int(res*ar), res))
            cv2.imwrite(savepath, img)
    
    if not keep_temp_stills:
        shutil.rmtree(temp_folder)
    
    print('Done. Took {:.4f}s'.format((time.time()-start)))
    return image_paths


#region - GEN SPRITESHEET ----------------------------------------------------------------------------------------------

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
    step = max(1, frame_count // (number_of_frames+1))
    
    # Determine optimal grid layout for spritesheet (aim for roughly square)
    cols = int(math.ceil(math.sqrt(number_of_frames)))
    rows = int(math.ceil(number_of_frames / cols))
    
    # Create blank spritesheet image
    spritesheet_width = cols * thumb_width
    spritesheet_height = rows * thumb_height
    spritesheet = np.zeros((spritesheet_height, spritesheet_width, 3), dtype=np.uint8)
    
    # Prepare VTT file content
    vtt_content = "WEBVTT\n\n"
    
    # Extract frames and build spritesheet
    frame, thumbnail = None, None
    for i in range(number_of_frames):
        print('\rextracting frame {}/{}'.format(i+1, number_of_frames), end='')
        # Calculate frame position and timestamp
        frame_pos = min((i+1) * step, frame_count - 1)
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
        start_time = i * (duration / number_of_frames)
        end_time = (i + 1) * (duration / number_of_frames)
        
        # Format times as HH:MM:SS.mmm
        start_time_str = _format_time(start_time)
        end_time_str = _format_time(end_time)
        
        # Add entry to VTT file
        vtt_content += f"{start_time_str} --> {end_time_str}\n"
        vtt_content += f"{filestem}.jpg#xywh={x},{y},{thumb_width},{thumb_height}\n\n"
    print()
    
    # Save spritesheet image
    spritesheet_path = os.path.join(output_dir, f"{filestem}.jpg")
    cv2.imwrite(spritesheet_path, spritesheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    
    # Save VTT file
    vtt_path = os.path.join(output_dir, f"{filestem}.vtt")
    with open(vtt_path, 'w') as f:
        f.write(vtt_content)
    
    # Release video capture
    cap.release()
    cv2.destroyAllWindows()  # (even if not using windows, just in case)
    del cap, spritesheet, frame, thumbnail
    
    return spritesheet_path, vtt_path



def _format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format for VTT files."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

