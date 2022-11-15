import cv2
import mediapipe as mp
from simple_facerec import SimpleFacerec
import numpy as np
import os


def position_data(lmlist):
    global wrist, thumb_tip, index_mcp, index_tip, midle_mcp, midle_tip, ring_mcp, ring_tip, pinky_tip
    wrist = (lmlist[0][0], lmlist[0][1])
    thumb_tip = (lmlist[4][0], lmlist[4][1])
    index_mcp = (lmlist[5][0], lmlist[5][1])
    index_tip = (lmlist[8][0], lmlist[8][1])
    midle_mcp = (lmlist[9][0], lmlist[9][1])
    midle_tip = (lmlist[12][0], lmlist[12][1])
    ring_mcp = (lmlist[13][0], lmlist[13][1])
    ring_tip = (lmlist[16][0], lmlist[16][1])
    pinky_tip = (lmlist[20][0], lmlist[20][1])


def draw_line(p1, p2, size=5):
    cv2.line(frame, p1, p2, (0, 255, 255), size)
    cv2.line(frame, p1, p2, (255, 255, 255), round(size / 2))


def calculate_distance(p1, p2):
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
    lenght = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1.0 / 2)
    return lenght


def transparent(targetImg, x, y, size=None):
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


absolute_path = os.path.dirname(__file__)
known_faces_folder = os.path.join(absolute_path, "known-faces/")
board_img = os.path.join(absolute_path, "images/name_board.png")

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images(known_faces_folder)
name_board = cv2.imread(board_img, -1)
board_size = (917 // 2, 508 // 2)
name_board = cv2.resize(name_board, board_size)

# Detect the hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=8)
mpDraw = mp.solutions.drawing_utils

# Load Camera and set the size of window
video = cv2.VideoCapture(0)
screen_res = (4096, 2160)
frame_res = (1920, 1080)
scale_width = screen_res[0] / frame_res[1]
scale_height = screen_res[1] / frame_res[0]
scale = min(scale_width, scale_height)
window_width = int(frame_res[1] * scale)
window_height = int(frame_res[0] * scale)

window_name = "Launch"
# cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, window_width, window_height)

img_1 = cv2.imread(os.path.join(absolute_path, 'images/magic_circle_ccw.png'), -1)
img_2 = cv2.imread(os.path.join(absolute_path, 'images/magic_circle_cw.png'), -1)
img_light = cv2.imread(os.path.join(absolute_path, 'images/starts_02.png'), -1)

deg = 0

while video.isOpened():
    video.set(cv2.CAP_PROP_FPS, 60)
    ret, frame = video.read()
    # frame = cv2.resize(frame, frame_res)
    frame = cv2.flip(frame, 1)

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        try:
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
            # cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
            bias = (x2 - x1) // 8
            frame_row = y1
            frame_col = x1 - bias

            # make the name board transparent
            b, g, r, a = cv2.split(name_board)
            overlay_color = cv2.merge((b, g, r))
            mask = cv2.medianBlur(a, 1)
            h, w, _ = overlay_color.shape
            roi = frame[
                frame_row : frame_row + board_size[1],
                frame_col - board_size[0] : frame_col,
            ]

            img1_bg = cv2.bitwise_and(
                roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask)
            )
            img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)

            frame[
                frame_row : frame_row + board_size[1],
                frame_col - board_size[0] : frame_col,
            ] = cv2.add(img1_bg, img2_fg)

        except:
            continue

    # Detect the Hands
    rgbimg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgbimg)

    # if result.multi_hand_landmarks and len(result.multi_hand_landmarks) >= 2:
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(hand.landmark):
                h, w, c = frame.shape
                coorx, coory = int(lm.x * w), int(lm.y * h)
                lmList.append([coorx, coory])
                # cv2.circle(frame, (coorx, coory),6,(50,50,255), -1)
            # mpDraw.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS)
            position_data(lmList)
            palm = calculate_distance(wrist, index_mcp)
            distance = calculate_distance(index_tip, pinky_tip)
            # palm = calculate_distance(wrist, ring_mcp)
            # distance = calculate_distance(ring_tip, ring_mcp)
            ratio = distance / palm
            # print(ratio)
            # if (0.8 >= ratio > 0.5):
            #     draw_line(wrist, thumb_tip)
            #     draw_line(wrist, index_tip)
            #     draw_line(wrist, midle_tip)
            #     draw_line(wrist, ring_tip)
            #     draw_line(wrist, pinky_tip)
            #     draw_line(thumb_tip, index_tip)
            #     draw_line(thumb_tip, midle_tip)
            #     draw_line(thumb_tip, ring_tip)
            #     draw_line(thumb_tip, pinky_tip)
            if ratio > 1.2:
                centerx = midle_mcp[0]
                centery = midle_mcp[1]
                shield_size = 3.0
                diameter = round(palm * shield_size)
                x1 = round(centerx - (diameter / 2))
                y1 = round(centery - (diameter / 2))
                h, w, c = frame.shape
                if x1 < 0:
                    x1 = 0
                elif x1 > w:
                    x1 = w
                if y1 < 0:
                    y1 = 0
                elif y1 > h:
                    y1 = h
                if x1 + diameter > w:
                    diameter = w - x1
                if y1 + diameter > h:
                    diameter = h - y1
                shield_size = diameter, diameter
                ang_vel = 2.0
                deg = deg + ang_vel
                if deg > 360:
                    deg = 0
                hei, wid, col = img_1.shape
                cen = (wid // 2, hei // 2)
                M1 = cv2.getRotationMatrix2D(cen, round(deg), 1.0)
                M2 = cv2.getRotationMatrix2D(cen, round(360 - deg), 1.0)
                rotated1 = cv2.warpAffine(img_1, M1, (wid, hei))
                rotated2 = cv2.warpAffine(img_2, M2, (wid, hei))
                if diameter != 0:
                    frame = transparent(rotated1, x1, y1, shield_size)
                    frame = transparent(rotated2, x1, y1, shield_size)

    # print(result)
    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break

video.release()
cv2.destroyAllWindows()
