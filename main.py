import cv2
import mediapipe as mp
from simple_facerec import SimpleFacerec
import os
import play_video
import play_video2
from utils import *
import sys

absolute_path = os.path.dirname(os.path.abspath(__file__))

######## Load variables ########
config_file = read_json(os.path.join(absolute_path, "conf/config.json"))

## limitions
authrized_faces_num = config_file["variables"]["authrized_faces_num"]
open_hand_limit = config_file["variables"]["open_hand_limit"]
ratio_limit = config_file["variables"]["ratio_limit"]
flash_rate = config_file["variables"]["flash_rate"]
rate_increment = config_file["variables"]["rate_increment"]
flash_color = config_file["variables"]["flash_color"]
max_num_hands = config_file["variables"]["max_num_hands"]

# Info board
info_board_size_rate = config_file["variables"]["info_board_size_rate"]
info_board_bias_rate = config_file["variables"]["info_board_bias_rate"]

## Window Name
window_name = config_file["variables"]["window_name"]

## The Camera
camera = config_file["variables"]["camera"]

######## Screen and Camera Resolutions ########
screen_res = config_file["variables"]["screen_res"]

# Load Camera and set the size of window
video = cv2.VideoCapture(camera)
ret, frame = video.read()

if not ret:
    print("Error: Cannot access webcam!")
    video.release()
    cv2.destroyAllWindows()
    exit()

frame_height, frame_width, _ = frame.shape

scale_width = screen_res[0] / frame_width
scale_height = screen_res[1] / frame_height
scale = min(scale_width, scale_height)

window_width = int(frame_width * scale)
window_height = int(frame_height * scale)

# cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, window_width, window_height)

######## Load the Images ########
# Shield related
shield_size = config_file["images"]["shield_size"]
shield_inside_image = config_file["images"]["shield_inside_image"]
shield_outside_image = config_file["images"]["shield_outside_image"]

img_outside = cv2.imread(os.path.join(absolute_path, shield_outside_image), -1)
img_inside = cv2.imread(os.path.join(absolute_path, shield_inside_image), -1)

deg = 0

# Encode faces from a folder
known_faces_folder = os.path.join(absolute_path, "known-faces/")
name_list = get_autherized_names(known_faces_folder)
sfr = SimpleFacerec()
sfr.load_encoding_images(known_faces_folder)


# Detect the hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
)
mpDraw = mp.solutions.drawing_utils

while video.isOpened():
    video.set(cv2.CAP_PROP_FPS, 60)
    ret, frame = video.read()
    show_hand_magic = False
    # frame = cv2.resize(frame, frame_res)
    frame = cv2.flip(frame, 1)

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    authrized_faces = [name for name in name_list if name in face_names]
    if len(authrized_faces) >= authrized_faces_num:
        show_hand_magic = True

    for face_loc, name in zip(face_locations, face_names):
        if name not in name_list:
            continue
        name_card = os.path.join(absolute_path, "name-cards/%s.png" % (name))
        name_board = cv2.imread(name_card, -1)
        height, width = name_board.shape[0], name_board.shape[1]
        board_size = (
            int(width * info_board_size_rate),
            int(height * info_board_size_rate),
        )
        name_board = cv2.resize(name_board, board_size)
        try:
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
            # cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
            bias_x = int((x2 - x1) * info_board_bias_rate)
            bias_y = int((y2 - y1) * info_board_bias_rate)
            frame_row = y1 - bias_y
            frame_col = x1 - bias_x

            # make the name board transparent
            b, g, r, a = cv2.split(name_board)
            overlay_color = cv2.merge((b, g, r))
            mask = cv2.medianBlur(a, 1)
            h, w, _ = overlay_color.shape
            roi = frame[
                frame_row - board_size[1] : frame_row,
                frame_col - board_size[0] // 2 : frame_col + board_size[0] // 2,
            ]

            img1_bg = cv2.bitwise_and(
                roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask)
            )
            img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)

            frame[
                frame_row - board_size[1] : frame_row,
                frame_col - board_size[0] // 2 : frame_col + board_size[0] // 2,
            ] = cv2.add(img1_bg, img2_fg)

        except:
            continue

    # Detect the Hands
    rgbimg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgbimg)

    open_hand_num = 0
    # if result.multi_hand_landmarks:
    if result.multi_hand_landmarks and show_hand_magic:
        for hand in result.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(hand.landmark):
                h, w, c = frame.shape
                coorx, coory = int(lm.x * w), int(lm.y * h)
                lmList.append([coorx, coory])
                # cv2.circle(frame, (coorx, coory),6,(50,50,255), -1)
            # mpDraw.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS)
            (
                wrist,
                thumb_tip,
                index_mcp,
                index_tip,
                midle_mcp,
                midle_tip,
                ring_mcp,
                ring_tip,
                pinky_tip,
            ) = position_data(lmList)
            # palm = calculate_distance(wrist, index_mcp)
            # distance = calculate_distance(index_tip, pinky_tip)
            palm = calculate_distance(wrist, midle_mcp)
            distance = calculate_distance(thumb_tip, pinky_tip)
            ratio = distance / palm

            if ratio >= ratio_limit:
                open_hand_num += 1
                centerx = midle_mcp[0]
                centery = midle_mcp[1]
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
                hei, wid, col = img_outside.shape
                cen = (wid // 2, hei // 2)
                M1 = cv2.getRotationMatrix2D(cen, round(deg), 1.0)
                M2 = cv2.getRotationMatrix2D(cen, round(360 - deg), 1.0)
                rotated1 = cv2.warpAffine(img_outside, M1, (wid, hei))
                rotated2 = cv2.warpAffine(img_inside, M2, (wid, hei))
                if diameter != 0:
                    frame = transparent(frame, rotated1, x1, y1, shield_size)
                    frame = transparent(frame, rotated2, x1, y1, shield_size)

    if open_hand_num >= open_hand_limit:
        cover_frame = frame.copy()
        cover_frame[:, :] = flash_color
        frame = cv2.addWeighted(frame, (1 - flash_rate), cover_frame, flash_rate, 0.0)
        flash_rate = flash_rate + rate_increment

    # print(result)
    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('p') or flash_rate >= 1:
        break

    if key == 27:
        exit()

video.release()
cv2.destroyAllWindows()

######## Play Video ########
video_path = config_file["videos"]["video_path"]
have_video_control = config_file["videos"]["have_video_control"]
video_window_title = config_file["videos"]["video_window_title"]

if have_video_control:
    app = play_video2.QApplication(sys.argv)
    window = play_video2.Window(video_path=video_path, window_title=video_window_title)
else:
    app = play_video.QApplication(sys.argv)
    window = play_video.Window(video_path=video_path, window_title=video_window_title)
window.show()
sys.exit(app.exec_())
