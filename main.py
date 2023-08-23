# Author : Nathaniel Thoma


# Libraries
import threading
import cv2
from deepface import DeepFace


# Variables
counter = 0                                    # counter used to prevent face recognizer to run every frame
ref_biden = cv2.imread("reference-biden.jpg")  # reference image of biden
ref_obama = cv2.imread("reference-obama.jpg")  # reference image of obama
ref_trump = cv2.imread("reference-trump.jpg")  # reference image of trump
biden_match = False                            # biden face match flag
obama_match = False                            # obama face match flag
trump_match = False                            # trump face match flag
green = (0, 255, 0)
red = (0, 0, 255)
font = cv2.FONT_HERSHEY_SIMPLEX


# Functions
def check_face(current_frame):  # check if any of the faces are recognizable
    global biden_match, obama_match, trump_match
    try:
        if DeepFace.verify(current_frame, ref_biden.copy())['verified']:    # if face is biden
            biden_match = True                                              # set the biden match flag to true
        elif DeepFace.verify(current_frame, ref_obama.copy())['verified']:  # if face is obama
            obama_match = True                                              # set the obama match flag to true
        elif DeepFace.verify(current_frame, ref_trump.copy())['verified']:  # if face is trump
            trump_match = True                                              # set the trump match flag to true
        else:                    # if none of the faces match
            biden_match = False  # set the biden flag to false
            obama_match = False  # set the obama flag to false
            trump_match = False  # set the trump flag to false
    except ValueError:       # if there is a value error exception
        biden_match = False  # set the biden flag to false
        obama_match = False  # set the obama flag to false
        trump_match = False  # set the trump flag to false


# Main
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # gets the capture object (camera)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # sets width of window
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # sets height of window

while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1

        if biden_match:
            cv2.putText(frame, "BIDEN!", (20, 450), font, 2, green, 3)
        elif obama_match:
            cv2.putText(frame, "OBAMA!", (20, 450), font, 2, green, 3)
        elif trump_match:
            cv2.putText(frame, "TRUMP!", (20, 450), font, 2, green, 3)
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450), font, 2, red, 3)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)  # process user input
    if key == ord("q"):   # if you press 'q'
        break             # break out of the loop

cv2.destroyAllWindows()  # when out of the loop, destroy windows and exit program
