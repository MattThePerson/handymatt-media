import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from xml.dom import minidom
from mutagen.mp4 import MP4


#region ### MKV ###

def getMetadataTags_MKV(mkv_file: str) -> dict[str, str]:
    """ Requires MKVToolNix. Gets metadata tags from mkv video using `mkvextract` """
    if not os.path.exists(mkv_file):
        raise FileNotFoundError('No such mkv file: "{}"'.format(mkv_file))
    xml_data = _get_mkv_xml_tags(mkv_file)
    tags = _extract_tags_from_xml(xml_data)
    return tags


def setMetadataTags_MKV(mkv_file: str, tags: dict[str, str]):
    """ Requires MKVToolNix. Sets metadata tags to mkv video using `mkvpropedit <mkv_file> --tags global:tags.xml` """
    if not os.path.exists(mkv_file):
        raise FileNotFoundError('No such mkv file: "{}"'.format(mkv_file))
    xml_data = _format_tags_into_xml(tags)
    _add_mkv_tags_from_xml(mkv_file, xml_data)


def addMetadataTags_MKV(mkv_file: str, tags: dict[str, str]):
    """ Requires MKVToolNix. Adds metadata tags to mkv video using `mkvpropedit <mkv_file> --tags global:tags.xml` """
    if not os.path.exists(mkv_file):
        raise FileNotFoundError('No such mkv file: "{}"'.format(mkv_file))
    # existing_tags = getMetadataTags_MKV(mkv_file)
    # _get
    existing_xml_data = _get_mkv_xml_tags(mkv_file)
    xml_data = _format_tags_into_xml_additive(tags, existing_xml_data)
    _add_mkv_tags_from_xml(mkv_file, xml_data)


def _add_mkv_tags_from_xml(mkv_file, xml_data):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=True) as temp:
        temp.write(xml_data)
        temp.flush()  # Ensure data is written before subprocess runs
        subprocess.run([
                "mkvpropedit", mkv_file,
                "--tags", f"all:{temp.name}"
            ],
            check=True,
            stdout=subprocess.DEVNULL,  # Suppress stdout
            stderr=subprocess.DEVNULL,  # Suppress stderr
        )

def _get_mkv_xml_tags(mkv_file: str) -> str:
    """ Uses `mkvextract` to extract tags which are in xml format """
    command = [
        'mkvextract', mkv_file, 'tags'
    ]
    out = subprocess.check_output(command).decode('utf-8')
    out = _clean_invalid_xml_characters(out)
    return out



#region ### MP4 ###

def getMetadataComment_MP4(mp4_file: str) -> str|None:
    """ DONT USE, SLOW AS SHIT """
    tags = MP4(mp4_file)
    comments = tags.get("\xa9cmt")
    if comments is not None:
        return comments[0]
    return None


def setMetadataComment_MP4(mp4_file: str, value: str):
    """  """
    tags = MP4(mp4_file)
    tags["\xa9cmt"] = [value]
    tags.save()


def getMetadataTitle_MP4(mp4_file: str) -> str|None:
    """ DONT USE, SLOW AS SHIT """
    tags = MP4(mp4_file)
    comments = tags.get("\xa9nam")
    if comments is not None:
        return comments[0]
    return None


def setMetadataTitle_MP4(mp4_file: str, value: str):
    """  """
    tags = MP4(mp4_file)
    tags["\xa9nam"] = [value]
    tags.save()


#region ### HLPERS ###


def _format_tags_into_xml(tags: dict[str, str]) -> str:
    """
    [By Deepseek] Converts a dictionary of tagName-tagString pairs into XML.
    Args:
        tags (dict): A dictionary where keys are tag names and values are tag strings.
    Returns:
        str: The generated XML as a formatted string.
    """
    # Create the root element
    root = ET.Element("Tags")

    # Iterate over the dictionary and create <Tag> elements
    for tag_name, tag_string in tags.items():
        tag = ET.SubElement(root, "Tag")
        simple = ET.SubElement(tag, "Simple")
        name = ET.SubElement(simple, "Name")
        name.text = tag_name
        string = ET.SubElement(simple, "String")
        string.text = tag_string

    # Convert the XML tree to a string
    xml_str = ET.tostring(root, encoding="unicode", method="xml")

    # Pretty-print the XML using minidom
    xml_pretty = minidom.parseString(xml_str).toprettyxml(indent="  ")

    return xml_pretty


def _format_tags_into_xml_additive(tags: dict[str, str], existing_xml: str) -> str:
    """ Adds tags formatted into xml to existing matroska xml tags """

    # Parse the existing XML
    root = ET.fromstring(existing_xml)

    # Add new tags to the existing XML
    for tag_name, tag_value in tags.items():
        tag = ET.SubElement(root, "Tag")
        simple = ET.SubElement(tag, "Simple")
        name = ET.SubElement(simple, "Name")
        name.text = tag_name
        string = ET.SubElement(simple, "String")
        string.text = tag_value

    # Convert the modified XML back to a string
    updated_xml = ET.tostring(root, encoding="unicode", method="xml")

    return updated_xml


def _extract_tags_from_xml(xml_str: str) -> dict[str, str]:
    """
    [By Deepseek] Converts XML in the specified format into a dictionary of tagName-tagString pairs.
    Args:
        xml_str (str): The XML string to parse.
    Returns:
        dict: A dictionary where keys are tag names and values are tag strings.
    """
    # Parse the XML string
    root = ET.fromstring(xml_str)

    # Initialize an empty dictionary
    tag_dict = {}

    # Iterate over each <Tag> element
    for tag in root.findall("Tag"):
        simple = tag.find("Simple")
        if simple is not None:
            name = simple.find("Name")
            string = simple.find("String")
            if name is not None and string is not None:
                tag_dict[name.text] = string.text

    return tag_dict


def _clean_invalid_xml_characters(xml_str):
    """
    Removes invalid XML characters from the input string using simple string operations.
    Specifically targets the form feed character (`&#12;`) and other invalid control characters.
    """
    # List of invalid XML 1.0 control characters (except \t, \n, \r)
    invalid_chars = [
        "\x00", "\x01", "\x02", "\x03", "\x04", "\x05", "\x06", "\x07", "\x08",  # 0x00-0x08
        "\x0B", "\x0C",  # 0x0B-0x0C
        "\x0E", "\x0F", "\x10", "\x11", "\x12", "\x13", "\x14", "\x15", "\x16", "\x17",  # 0x0E-0x17
        "\x18", "\x19", "\x1A", "\x1B", "\x1C", "\x1D", "\x1E", "\x1F",  # 0x18-0x1F
        "\x7F"  # 0x7F
    ]
    xml_str = xml_str.replace(r"&#12;", "")
    xml_str = xml_str.replace('&', "and")
    for char in invalid_chars:
        xml_str = xml_str.replace(char, "")

    return xml_str