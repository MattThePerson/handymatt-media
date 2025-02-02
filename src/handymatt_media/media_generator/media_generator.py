import os
import subprocess
import shlex
import os
from pathlib import Path
import time
import shutil

import cv2
from nudenet import NudeDetector

from .lib.stills import extract_stills_from_video, addDetectionsToImage, get_detections_score, floodingMethod


# GENERATE VIDEO TEASER
def generateVideoTeaser(input_path, output_dir, savename, abs_amount_mode=False, n=10, jump=300, clip_len=1.3, start_perc=5, end_perc=95, keep_clips=False, skip=1, smallSize=False, quit=True):
    if not os.path.exists(input_path):
        print("ERROR: Path doesn't exist [{}]".format(input_path))
        return None
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
        command = f'ffmpeg -ss {time} -i "{input_path}" -t 00:00:{clip_len} -y -map 0:0 -map 0:1? -c:v libx264 -v quiet -stats'
        if smallSize:
            command += f' -vf scale={smallRes}'
        command += f' "{tempname}"'
        subprocess.run(shlex.split(command))
        tempnames.append(tempname)
        # -v quiet -stats
    print()
    print("Concatenating {} clips ...".format(len(tempnames)))
    savepath = _concatClips(savepath, savename, tempnames)
    if not keep_clips:
        for clip in tempnames:
            os.remove(clip)
    print("Done.")
    return savepath


def _concatClips(savepath, savename, clips):
    clips_command = []
    filter_start = '-filter_complex "'
    for i, clip in enumerate(clips):
        clips_command.append("-i")
        clips_command.append(clip)
        filter_start = filter_start + f" [{i}:v]"
    filter_end = ' concat=n={}:v=1 [v]"'.format(len(clips))
    filter_command = filter_start + filter_end
    end = '-map "[v]" -y'
    #     filter_start = filter_start + f" [{i}:v] [{i}:a]"
    # filter_end = ' concat=n={}:v=1:a=1 [v] [a]"'.format(len(clips))
    # filter_command = filter_start + filter_end
    # end = '-map "[v]" -map "[a]" -y'
    command = ["ffmpeg"] + clips_command + _getCommandTerms(filter_command) + _getCommandTerms(end) + [savepath]
    print('running command [{}]'.format(command))
    _ = subprocess.run(command)
    # -v quiet -stats
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



if __name__ == '__main__':
    ...