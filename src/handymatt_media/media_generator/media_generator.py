import os
import time
import shutil
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


