""" Preview media generators that only use ffmpeg via subprocess """
import os
import subprocess
from pathlib import Path


#region - VIDEO TEASER -------------------------------------------------------------------------------------------------

def generateVideoTeaser(
        input_path: str,
        output_dir: str,
        savename: str,
        abs_amount_mode=False,
        n=10,
        jump=300,
        clip_len=1.3,
        start_perc=5,
        end_perc=95,
        keep_clips=False,
        skip=1,
        small_resolution=False,
    ) -> str:
    """
    (ffmpeg) Extracts n short clips from video and concats them together
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError("Video path doesn't exist: {}".format(input_path))

    print("Generating preview for video:", input_path)

    duration_sec = _get_video_duration(input_path)
    start_sec =   duration_sec * start_perc/100
    end_sec =     duration_sec * end_perc/100
    
    if abs_amount_mode:
        jump = (end_sec - start_sec) / n
    
    # compute clip times
    curr_sec = start_sec
    skipCount = skip
    times = []
    while curr_sec < end_sec:
        skipCount -= 1
        if skipCount == 0:
            times.append(_formatSeconds(curr_sec))
            skipCount = skip
        curr_sec += jump
    
    # extract clips
    tempnames = []
    for idx, time_fmt in enumerate(times):
        print("\rExtracting clip ({}/{}) at time [{}]".format(idx+1, len(times), time_fmt), end='')
        tempname = os.path.join( output_dir, f'temp_{idx+1}.mp4' )
        _extract_clip(
            input_path,
            tempname,
            time_fmt,
            clip_len,
            small_resolution,
        )
        tempnames.append(tempname)
    
    print("\nConcatenating {} clips ...".format(len(tempnames)))
    savepath = os.path.join( output_dir, savename )
    _concat_clips(savepath, tempnames)
    if not keep_clips:
        for clip in tempnames:
            os.remove(clip)
    print("Done.")
    return savepath



def _extract_clip(input_path: str, save_path: str, time: str, clip_len: float, small_resolution: bool=False) -> int:
    
    command = [
        'ffmpeg', '-ss', time,
        '-i', input_path,
        '-t', f'00:00:{clip_len}',
        '-map', '0:v:0', '-an',
        '-c:v', 'libx264',
        '-v', 'error',
        save_path, '-y',
    ]
    if small_resolution:
        command = command + ['-vf', 'scale=640:360']
    # print(' '.join(command))
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("FFmpeg returncode = {} while processing video \"{}\"".format(result.returncode, input_path))
    if "Error" in result.stderr or "Invalid" in result.stderr:
        raise RuntimeError("FFmpeg returned 0 but has stderr for video \"{}\":\n{}".format(input_path, result.stderr[:1000]))
    
    return 0



def _concat_clips(savepath: str, clips: list[str]) -> int:
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
    if not os.path.exists(savepath):
        raise FileNotFoundError('Video not found after concatenation:', savepath)
    return 0


def _get_video_duration(input_path: str):
    command = [
        'ffmpep',
        '-v', 'error',
        'show_entries', 'format=duration', '-of',
        'default=noprint_wrappers=1:nokey=1',
        input_path,
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return int(float(result.stdout))


#region - GEN SPRITESHEET ----------------------------------------------------------------------------------------------

def generateVideoSpritesheet_ffmpeg(
        video_path: str,
        output_dir: str,
        filestem: str='spritesheet',
        number_of_frames: int=100,
        height: int=300,
        keep_tempfiles: bool=False,
        verbose: bool=False,
    ):
    """ (ffmpeg) (HUOM: This shit is slow!!) For a given video, will generate a spritesheet of seek thumbnails (preview thumbnails) as well as .vtt file. """
    import math
    import numpy as np
    
    if not os.path.exists(video_path):
        raise FileNotFoundError('Video doesnt exist:', video_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    tmp_dir = os.path.join(output_dir, 'tmp_frames')
    os.makedirs(tmp_dir, exist_ok=True)

    # get duration and thumb size
    width, height_src, duration = _get_video_res_and_duration(video_path)
    aspect_ratio = width / height_src
    thumb_width = int(height * aspect_ratio)
    thumb_height = height

    # Extract n frames evenly spaced
    interval = duration / number_of_frames
    extract_pattern = os.path.join(tmp_dir, 'frame_%03d.jpg')

    print(f'[GEN] Extracting frames using interval:', interval)
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vf', f'fps=1/{interval},scale={thumb_width}:{thumb_height}',
        extract_pattern,
    ]
    if not verbose:
        cmd += ['-v', 'error', '-stats']
    subprocess.run(cmd, check=True)

    # Create spritesheet using ffmpeg tile filter
    cols = math.ceil(math.sqrt(number_of_frames))
    rows = math.ceil(number_of_frames / cols)
    spritesheet_path = os.path.join(output_dir, f"{filestem}.jpg")
    print(f'[GEN] Combining into spritesheet with (cols, rows) = ({cols}, {rows})')
    cmd = [
        'ffmpeg', '-y', '-i', os.path.join(tmp_dir, 'frame_%03d.jpg'),
        '-filter_complex', f'tile={cols}x{rows}',
        spritesheet_path,
    ]
    if not verbose:
        cmd += ['-v', 'error', '-stats']
    subprocess.run(cmd, check=True)

    # Generate VTT file
    vtt_path = os.path.join(output_dir, f"{filestem}.vtt")
    vtt = "WEBVTT\n\n"
    for i in range(number_of_frames):
        x = (i % cols) * thumb_width
        y = (i // cols) * thumb_height
        start = _format_time(i * interval)
        end = _format_time((i + 1) * interval)
        vtt += f"{start} --> {end}\n{filestem}.jpg#xywh={x},{y},{thumb_width},{thumb_height}\n\n"

    with open(vtt_path, 'w') as f:
        f.write(vtt)

    # Cleanup
    if not keep_tempfiles:
        for f in Path(tmp_dir).glob("*.jpg"):
            f.unlink()
        os.rmdir(tmp_dir)

    return spritesheet_path, vtt_path



def _get_video_res_and_duration(video_path: str) -> ...:
    # Get video duration and dimensions
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    str_values = result.stdout.strip().split('\n')
    width, height, fps_num, duration = [ float(eval(v)) for v in str_values ] # map(lambda x: float(eval(x)), str_values)
    return width, height, duration



def _format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format for VTT files."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"




#region - CREATE GIF ---------------------------------------------------------------------------------------------------

def createGif(videopath, savepath, start_time_sec, gif_duration=7, resolution=720, fps=15):
    print("Creating gif for video at path: \"{}\"".format(videopath))
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



#region - HELPERS ------------------------------------------------------------------------------------------------------

def _formatSeconds(sec: float) -> str:
    h = int(sec / 3600)
    sec -= h * 3600
    m = int(sec / 60)
    sec -= m*60
    s = int(sec)
    return f"{h}:{m}:{s}"

