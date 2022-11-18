import cv2
from simple_facerec import SimpleFacerec
import face_recognition as fr
import os
from utils import *

absolute_path = os.path.dirname(os.path.abspath(__file__))

######## Load variables ########
config_file = read_json(os.path.join(absolute_path, "conf/config.json"))

## Window Name
window_name = config_file["variables"]["window_name"]

## The Camera
camera = config_file["variables"]["camera"]
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

######## Load the Images ########

# Encode faces from a folder
known_faces_folder = os.path.join(absolute_path, "known-faces/")
name_list = get_autherized_names(known_faces_folder)
sfr = SimpleFacerec()
sfr.load_encoding_images(known_faces_folder)

video = cv2.VideoCapture(0)

face_cover_img = cv2.imread(os.path.join(absolute_path, "images/tech_eye.png"), -1)
height, width = face_cover_img.shape[0], face_cover_img.shape[1]
print(height, width)
info_board_size_rate = 0.5
board_size = (
    int(width * info_board_size_rate),
    int(height * info_board_size_rate),
)
face_cover_img = cv2.resize(face_cover_img, board_size)

while video.isOpened():
    video.set(cv2.CAP_PROP_FPS, 60)
    ret, frame = video.read()
    # frame = cv2.resize(frame, frame_res)
    frame = cv2.flip(frame, 1)

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    authrized_faces = [name for name in name_list if name in face_names]

    for face_loc, name in zip(face_locations, face_names):
        if name not in name_list:
            continue

        try:
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
            # cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
            face_area = frame[y1:y2, x1:x2]

            landmarks = fr.face_landmarks(face_area)
            if landmarks:
                right_eye = landmarks[0]["right_eye"]
                right_eye_point_locations = [(x1 + x, y1 + y) for x, y in right_eye]

                x = [p[0] for p in right_eye_point_locations]
                y = [p[1] for p in right_eye_point_locations]

                right_eye_x, right_eye_y = (
                    sum(x) // len(right_eye_point_locations),
                    sum(y) // len(right_eye_point_locations),
                )

                left_eye = landmarks[0]["left_eye"]
                left_eye_point_locations = [(x1 + x, y1 + y) for x, y in left_eye]

                x = [p[0] for p in left_eye_point_locations]
                y = [p[1] for p in left_eye_point_locations]

                left_eye_x, left_eye_y = (
                    sum(x) // len(left_eye_point_locations),
                    sum(y) // len(left_eye_point_locations),
                )

                eye_x, eye_y = (left_eye_x + right_eye_x) // 2, (left_eye_y + right_eye_y) // 2 


                img_height, img_weight, _ = face_cover_img.shape

                roi = frame[
                    right_eye_y - img_height // 2 : right_eye_y + img_height // 2,
                    right_eye_x - img_weight // 2 : right_eye_x + img_weight // 2,
                ]

                # cover = cv2.resize(face_cover_img, (img_height // 2, img_weight // 2))
                # make the name board transparent
                b, g, r, a = cv2.split(face_cover_img)
                overlay_color = cv2.merge((b, g, r))
                mask = cv2.medianBlur(a, 1)
                h, w, _ = overlay_color.shape

                img1_bg = cv2.bitwise_and(
                    roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask)
                )
                img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)

                frame[
                    right_eye_y - img_height // 2 : right_eye_y + img_height // 2,
                    right_eye_x - img_weight // 2 : right_eye_x + img_weight // 2,
                ] = cv2.add(img1_bg, img2_fg)
        except:
            continue

    # print(result)
    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1)

    if key == 27:
        break

video.release()
cv2.destroyAllWindows()
