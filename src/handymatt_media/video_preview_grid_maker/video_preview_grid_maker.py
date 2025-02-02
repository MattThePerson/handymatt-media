import cv2
import os
import numpy as np

# 
def add_image_to_image(img_add, img_base, pos=(0,0)):
    img_base[pos[0]:pos[0]+img_add.shape[0], pos[1]:pos[1]+img_add.shape[1]] = img_add
    return img_base

# 
def add_text_to_image(img, text, pos=(50, 50), font_size=1, font_color=(50, 50, 50), thickness=1, shadow=False):
    if type(font_color) != tuple:
        font_color = (font_color, font_color, font_color)
    if pos[0] < 0:  pos = (img.shape[1] + pos[0], pos[1])
    if pos[1] < 0:  pos = (pos[0], img.shape[0] + pos[1])
    if shadow:
        off = 4
        img = cv2.putText(img, text, (pos[0]+2*off, pos[1]+off), cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, (0, 0, 0), int(thickness*1.3), cv2.LINE_AA)
    img = cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                font_size, font_color, thickness, cv2.LINE_AA)
    return img

# 
def frames_to_time(frames, fps):
    seconds = int(frames / fps)
    h = int(seconds / 3600)
    seconds -= h * 3600
    m = int(seconds / 60)
    s = seconds - m * 60
    return f"{h:02d}:{m:02d}:{s:02d}"

# 
def generate_header(width, filepath, cap, resolution=None, background=255):
    filestats = os.stat(filepath)
    size_mb = filestats.st_size / (1024 * 1024)
    #print(f'File Size in MegaBytes is {}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frames_to_time(amount_of_frames, fps)
    lines = [
        "FILENAME: " + os.path.basename(filepath),
        "DURATION: " + duration + "  SIZE MB: " + str(int(size_mb)),
        "RESOLUTION: {}x{}".format(*resolution) + "  FPS: " + str(int(fps)) + "  BITRATE: " + str(int(size_mb*1024*8/(amount_of_frames/fps))) + " kbps",
    ]
    padding = 5
    shift =  20
    header_thickness = 2 * padding + len(lines) * shift
    header = np.ones((header_thickness, width, 3), dtype=np.uint8) * background
    for i, line in enumerate(lines):
        pos = ( padding, 15 + padding + i * shift )
        header = add_text_to_image(header, line, pos, thickness=1, font_size=0.5)
    return header

# 
def generatePreviewGrid(filepath, savefolder, savename=None, grid=(4,5), width=800, start_perc=5, end_perc=95, font_size=18, background=255):

    if not os.path.exists(filepath):
        print("ERROR: filepath doesn't exist,", filepath)
        return
    if not os.path.exists(savefolder):
        try:
            os.mkdir(savefolder)
            print("Made save folder:", savefolder)
        except:
            print("ERROR: Savefolder doesn't exist and mkdir failed.\n   savefolder:", savefolder)
            return
    
    print("Generating preview grid for:", os.path.basename(filepath))
    
    # init capture
    cap = cv2.VideoCapture(filepath)
    amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frames_per_second = cap.get(cv2.CAP_PROP_FPS)
    frames_to_read = grid[0] * grid[1]

    start_frame = int(start_perc * amount_of_frames / 100)
    end_frame = int(end_perc * amount_of_frames / 100)
    step = int( (end_frame - start_frame) / (frames_to_read-1) )
    frames_captured = 0
    grid_x, grid_y = 0, 0

    # determine thumb params
    padding = 2
    _, first_frame = cap.read()
    resolution = (first_frame.shape[1], first_frame.shape[0])
    thumb_width_flt = (width-padding) / grid[0] - padding
    thumb_height_flt = thumb_width_flt * first_frame.shape[0] / first_frame.shape[1]
    thumb_width, thumb_height = int(thumb_width_flt), int(thumb_height_flt)

    # make header
    header = generate_header(width, filepath, cap, resolution=resolution, background=background)
    header_thickness = header.shape[0]

    # make base image
    image_height =  + int( header_thickness + padding + (thumb_height_flt + padding) * grid[1] )
    image = np.ones((image_height, width, 3), dtype=np.uint8) * background
    image = add_image_to_image(header, image)

    while (True):
        frame_to_read = start_frame + frames_captured * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_read)
        ret, frame = cap.read()

        if ret:
            print("\rHandling frame ({}/{})".format(frames_captured+1, frames_to_read), end='')
            frame_time = frames_to_time(frame_to_read, frames_per_second)
            frame = add_text_to_image(frame, frame_time, shadow=True, pos=(15, -20), thickness=6, font_size=2.7, font_color=(255))

            frame = cv2.resize(frame, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA )
            pos = (
                int( header_thickness + padding + grid_y * (thumb_height_flt + padding) ),
                int( padding + grid_x * (thumb_width_flt + padding) )
            )
            #print("Putting image at pos:", pos)
            image = add_image_to_image(frame, image, pos)

            frames_captured += 1
            grid_x += 1
            if grid_x >= grid[0]:
                grid_x = 0
                grid_y += 1
            if frames_captured >= frames_to_read:
                ret = False
            #if frames_captured >= 2:
            #    break
            
        if not ret:
            break
    
    if frames_captured < frames_to_read:
        print("ERROR: Unable to extract all frames")
        return

    if savename == None:
        savename = "{}_preview_({}x{}).jpg".format(os.path.basename(filepath), grid[0], grid[1])
    savepath = os.path.join(savefolder, savename)
    print("\nSaving image to:", savepath)
    cv2.imwrite(savepath, image)
    #cv2.imshow('image', image)
    #cv2.waitKey(0)
    
    cap.release()
    cv2.destroyAllWindows()

    return savepath