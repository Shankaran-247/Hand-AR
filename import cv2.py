import cv2
import mediapipe as mp
import numpy as np
import math


# ==============================
# MEDIAPIPE SETUP
# ==============================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# ==============================
# CAMERA
# ==============================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera")
    exit()


# ==============================
# AR OBJECT
# ==============================

cube_x = 320
cube_y = 240
cube_size = 100

selected = False


# ==============================
# HELPER FUNCTIONS
# ==============================

def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def get_gesture(landmarks):

    # Landmark positions
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]

    index_pip = landmarks[6]
    middle_tip = landmarks[12]
    middle_pip = landmarks[10]

    ring_tip = landmarks[16]
    ring_pip = landmarks[14]

    pinky_tip = landmarks[20]
    pinky_pip = landmarks[18]

    # --------------------------------
    # PINCH
    # --------------------------------

    pinch_distance = distance(
        thumb_tip,
        index_tip
    )

    if pinch_distance < 35:
        return "PINCH"


    # --------------------------------
    # FINGERS UP
    # --------------------------------

    index_up = index_tip[1] < index_pip[1]
    middle_up = middle_tip[1] < middle_pip[1]
    ring_up = ring_tip[1] < ring_pip[1]
    pinky_up = pinky_tip[1] < pinky_pip[1]


    # --------------------------------
    # OPEN PALM
    # --------------------------------

    if index_up and middle_up and ring_up and pinky_up:
        return "OPEN PALM"


    # --------------------------------
    # POINT
    # --------------------------------

    if index_up and not middle_up and not ring_up and not pinky_up:
        return "POINT"


    # --------------------------------
    # FIST
    # --------------------------------

    if not index_up and not middle_up and not ring_up and not pinky_up:
        return "FIST"


    return "UNKNOWN"


# ==============================
# DRAW AR OBJECT
# ==============================

def draw_cube(frame, x, y, size, selected):

    # Front face
    p1 = (x, y)
    p2 = (x + size, y)
    p3 = (x + size, y + size)
    p4 = (x, y + size)

    # Back face
    offset = int(size * 0.25)

    b1 = (x + offset, y - offset)
    b2 = (x + size + offset, y - offset)
    b3 = (x + size + offset, y + size - offset)
    b4 = (x + offset, y + size - offset)

    # Front
    cv2.rectangle(
        frame,
        p1,
        p3,
        (255, 255, 255),
        2
    )

    # Back
    cv2.rectangle(
        frame,
        b1,
        b3,
        (255, 255, 255),
        2
    )

    # Connecting lines
    cv2.line(frame, p1, b1, (255, 255, 255), 2)
    cv2.line(frame, p2, b2, (255, 255, 255), 2)
    cv2.line(frame, p3, b3, (255, 255, 255), 2)
    cv2.line(frame, p4, b4, (255, 255, 255), 2)

    # Selection box
    if selected:

        cv2.rectangle(
            frame,
            (x - 10, y - 10),
            (x + size + 10, y + size + 10),
            (0, 255, 255),
            3
        )


# ==============================
# MAIN LOOP
# ==============================

while True:

    success, frame = cap.read()

    if not success:
        break


    # Mirror camera
    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape


    # Convert BGR → RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Detect hands
    results = hands.process(rgb)


    gesture = "NO HAND"
    hand_x = None
    hand_y = None


    if results.multi_hand_landmarks:

        # Use first detected hand
        hand = results.multi_hand_landmarks[0]

        # Draw hand skeleton
        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )


        # Convert landmarks to pixels
        landmarks = []

        for landmark in hand.landmark:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            landmarks.append((x, y))


        # Gesture
        gesture = get_gesture(
            landmarks
        )


        # Index finger position
        hand_x = landmarks[8][0]
        hand_y = landmarks[8][1]


        # ==========================
        # PINCH INTERACTION
        # ==========================

        if gesture == "PINCH":

            # Check whether hand is over cube

            if (
                cube_x - 20 < hand_x <
                cube_x + cube_size + 20
                and
                cube_y - 20 < hand_y <
                cube_y + cube_size + 20
            ):

                selected = True

        else:

            selected = False


        # ==========================
        # MOVE OBJECT
        # ==========================

        if selected:

            cube_x = hand_x - cube_size // 2
            cube_y = hand_y - cube_size // 2


    # ==============================
    # DRAW AR OBJECT
    # ==============================

    draw_cube(
        frame,
        cube_x,
        cube_y,
        cube_size,
        selected
    )


    # ==============================
    # HUD
    # ==============================

    # Top HUD
    cv2.rectangle(
        frame,
        (20, 20),
        (350, 125),
        (20, 20, 20),
        -1
    )

    cv2.rectangle(
        frame,
        (20, 20),
        (350, 125),
        (255, 255, 255),
        1
    )


    cv2.putText(
        frame,
        "HAND-TRACKING AR",
        (40, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "GESTURE:",
        (40, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1
    )


    cv2.putText(
        frame,
        gesture,
        (140, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )


    # Tracking status
    status = "TRACKING ACTIVE" if results.multi_hand_landmarks else "SEARCHING HAND..."

    cv2.putText(
        frame,
        status,
        (40, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )


    # ==============================
    # GESTURE HELP
    # ==============================

    cv2.putText(
        frame,
        "PINCH = SELECT / MOVE",
        (20, height - 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        "POINT = SELECT",
        (20, height - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        "FIST = RESET",
        (20, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    # ==============================
    # RESET WITH FIST
    # ==============================

    if gesture == "FIST":

        cube_x = width // 2 - cube_size // 2
        cube_y = height // 2 - cube_size // 2


    # ==============================
    # EXIT
    # ==============================

    cv2.putText(
        frame,
        "Press Q to exit",
        (width - 150, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1
    )


    cv2.imshow(
        "Hand Tracking AR",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ==============================
# CLEANUP
# ==============================

cap.release()
cv2.destroyAllWindows()
hands.close()