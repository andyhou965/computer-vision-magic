import glob
import os
import cv2
import json


def get_autherized_names(path):
    # Load Images
    images_path = glob.glob(os.path.join(path, "*.*"))
    name_list = []
    for img_path in images_path:
        basename = os.path.basename(img_path)
        filename, _ = os.path.splitext(basename)
        name_list.append(filename)

    return name_list


def position_data(lmlist):
    wrist = (lmlist[0][0], lmlist[0][1])
    thumb_tip = (lmlist[4][0], lmlist[4][1])
    index_mcp = (lmlist[5][0], lmlist[5][1])
    index_tip = (lmlist[8][0], lmlist[8][1])
    midle_mcp = (lmlist[9][0], lmlist[9][1])
    midle_tip = (lmlist[12][0], lmlist[12][1])
    ring_mcp = (lmlist[13][0], lmlist[13][1])
    ring_tip = (lmlist[16][0], lmlist[16][1])
    pinky_tip = (lmlist[20][0], lmlist[20][1])

    return (
        wrist,
        thumb_tip,
        index_mcp,
        index_tip,
        midle_mcp,
        midle_tip,
        ring_mcp,
        ring_tip,
        pinky_tip,
    )


def calculate_distance(p1, p2):
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
    lenght = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1.0 / 2)
    return lenght


def transparent(frame, targetImg, x, y, size=None):
    if size is not None:
        targetImg = cv2.resize(targetImg, size)

    newFrame = frame.copy()
    b, g, r, a = cv2.split(targetImg)
    overlay_color = cv2.merge((b, g, r))
    mask = cv2.medianBlur(a, 1)
    h, w, _ = overlay_color.shape
    roi = newFrame[y : y + h, x : x + w]

    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)
    newFrame[y : y + h, x : x + w] = cv2.add(img1_bg, img2_fg)

    return newFrame


def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
