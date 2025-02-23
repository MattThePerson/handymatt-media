from PIL import Image, PngImagePlugin
import piexif
import json

__all__ = ['ImageMetadata_WriteJson', 'ImageMetadata_ReadJson', 'ImageMetadata_Remove']


def ImageMetadata_WriteJson(img: str, metadata: dict) -> None:
    '''Writes metadata in a json format to the metadata description of an image. 
    Handles jpg and png separately.'''
    ext = img.split('.')[-1].lower()
    if ext in ['jpg', 'jpeg']:
        write_description_jpg(img, metadata)
    elif ext in ['png']:
        write_description_png(img, metadata)
    else:
        print(f"ERROR: {ext} not a supported image extension")


def ImageMetadata_ReadJson(img: str) -> dict | None:
    '''Reads the metadata description of an image and attempts to parse it as json.
    Handles jpg in png separately.'''
    ...
    desc_str = ""
    ext = img.split('.')[-1].lower()
    if ext in ['jpg', 'jpeg']:
        desc_str = read_description_jpg(img)
    elif ext in ['png']:
        desc_str = read_description_png(img)
    else:
        print(f"ERROR: {ext} not a supported image extension")
        return None
    
    if desc_str == None:
        return None
    
    try:
        data = json.loads(desc_str)
        if isinstance(data, dict):
            return data
        return None
    except:
        print('ERROR: Unable to parse description as json for "{}"\nDESC:"{}"'.format(img, desc_str))
        return None

def ImageMetadata_Remove(img: str):
    '''Removes description attribute from image metadata. Handles jpg in png separately.'''
    ext = img.split('.')[-1].lower()
    if ext in ['jpg', 'jpeg']:
        remove_description_jpg(img)
    elif ext in ['png']:
        remove_description_png(img)
    else:
        print(f"ERROR: {ext} not a supported image extension")
        return None

#Write JSON string to PNG description.
def write_description_png(file_path: str, json_data: dict):
    image = Image.open(file_path)
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text("Description", json.dumps(json_data))
    image.save(file_path, pnginfo=png_info)

#Read JSON string from PNG description.
def read_description_png(file_path: str) -> str | None:
    image = Image.open(file_path)
    desc_str = image.info.get("Description", None)
    return desc_str

#Write JSON string to JPG description using EXIF.
def write_description_jpg(file_path: str, json_data: dict):
    exif_dict = piexif.load(file_path)
    json_string = json.dumps(json_data)
    exif_dict["0th"][piexif.ImageIFD.ImageDescription] = json_string.encode('utf-8')
    exif_bytes = piexif.dump(exif_dict)
    Image.open(file_path).save(file_path, exif=exif_bytes)

#Read JSON string from JPG description using EXIF.
def read_description_jpg(file_path: str) -> str | None:
    exif_dict = piexif.load(file_path)
    desc_str = exif_dict["0th"].get(piexif.ImageIFD.ImageDescription)
    if desc_str == None:
        return None
    return desc_str.decode('utf-8')

#Remove the description metadata from a PNG file.
def remove_description_png(file_path):
    image = Image.open(file_path)
    png_info = PngImagePlugin.PngInfo()  # Create an empty metadata object
    image.save(file_path, pnginfo=png_info)    # Save without metadata

#Remove the description metadata from a JPG file.
def remove_description_jpg(file_path):
    exif_dict = piexif.load(file_path)
    # Remove the description key if it exists
    if piexif.ImageIFD.ImageDescription in exif_dict["0th"]:
        del exif_dict["0th"][piexif.ImageIFD.ImageDescription]
    exif_bytes = piexif.dump(exif_dict)
    Image.open(file_path).save(file_path, exif=exif_bytes)



if __name__ == '__main__':
    jpgs = [
        'PNG_transparency_demonstration_1.png',
        'Gf487ivawAYMdMR.jpeg',
        '1869414850130284799_1.jpg',
    ]

    # Example usage
    metadata = {
        "author": "John Doe",
        "description": "Sample metadata",
        "tags": ["example", "metadata"]
    }
    img = jpgs[2]
    metadata['fn'] = img
    ImageMetadata_WriteJson(img, {})
    # ImageMetadata_Remove(img)
    data = ImageMetadata_ReadJson(img)
    print('  IMAGE "{}"'.format(img))
    if data != None:
        for k, v in data.items():
            print(k, v)
    else:
        print(data)
