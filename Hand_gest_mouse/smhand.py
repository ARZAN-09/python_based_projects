import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

mirror_mode = True
show_gestures = True
scroll_mode = False
click_mode = True

screen_w, screen_h = pyautogui.size()
prev_x, prev_y = screen_w // 2, screen_h // 2
smooth_factor = 0.2

click_threshold = 0.05
click_cooldown = 0.4
last_click_time = 0

scroll_threshold = 0.08
scroll_cooldown = 0.15
last_scroll_time = 0

prev_time = time.time()
activity_text = "Idle"
gesture_info = ""

def calculate_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def get_finger_status(landmarks):
    fingers = []
    
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]
    if thumb_tip.x > thumb_mcp.x:
        fingers.append(1)
    else:
        fingers.append(0)
    
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    
    for i in range(4):
        if landmarks[tips[i]].y < landmarks[pips[i]].y:
            fingers.append(1)
        else:
            fingers.append(0)
    
    return fingers

def draw_ui_panel(img, x, y, width, height, alpha=0.7):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (40, 40, 40), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x + width, y + height), (100, 100, 100), 2)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    if mirror_mode:
        image = cv2.flip(image, 1)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_image)
    h, w, _ = image.shape

    if not result.multi_hand_landmarks:
        activity_text = "No Hand"
        gesture_info = "Show your hand to camera"
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            if show_gestures:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            thumb_tip = hand_landmarks.landmark[4]
            
            fingers = get_finger_status(hand_landmarks.landmark)
            fingers_up = sum(fingers)
            
            cam_x = index_tip.x
            cam_y = index_tip.y
            
            expanded_x = np.interp(cam_x, [0.15, 0.85], [0, screen_w])
            expanded_y = np.interp(cam_y, [0.15, 0.85], [0, screen_h])
            
            smooth_x = int(prev_x + (expanded_x - prev_x) * smooth_factor)
            smooth_y = int(prev_y + (expanded_y - prev_y) * smooth_factor)
            
            smooth_x = max(50, min(screen_w - 50, smooth_x))
            smooth_y = max(50, min(screen_h - 50, smooth_y))

            try:
                pyautogui.moveTo(smooth_x, smooth_y)
                prev_x, prev_y = smooth_x, smooth_y
                activity_text = "Tracking"
            except:
                pass

            current_time = time.time()
            
            if click_mode:
                click_distance = calculate_distance(index_tip, thumb_tip)
                if click_distance < click_threshold and (current_time - last_click_time) > click_cooldown:
                    try:
                        pyautogui.click()
                        activity_text = "CLICK"
                        last_click_time = current_time
                    except:
                        pass

            if scroll_mode and (current_time - last_scroll_time) > scroll_cooldown:
                index_middle_distance = calculate_distance(index_tip, middle_tip)
                
                if index_middle_distance < scroll_threshold:
                    if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
                        if index_tip.y < 0.4:
                            try:
                                pyautogui.scroll(5)
                                activity_text = "SCROLL UP"
                                last_scroll_time = current_time
                            except:
                                pass
                        elif index_tip.y > 0.6:
                            try:
                                pyautogui.scroll(-5)
                                activity_text = "SCROLL DOWN"
                                last_scroll_time = current_time
                            except:
                                pass

            gesture_info = f"Fingers: {fingers_up} | "
            if scroll_mode:
                gesture_info += "Peace sign + position to scroll"
            else:
                gesture_info += "Pinch to click"

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    
    draw_ui_panel(image, 10, 10, 350, 160)
    
    cv2.putText(image, f'FPS: {int(fps)}', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(image, f'Status: {activity_text}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(image, f'{gesture_info}', (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    mode_text = ""
    if mirror_mode:
        mode_text += "Mirror "
    if scroll_mode:
        mode_text += "Scroll "
    if click_mode:
        mode_text += "Click "
    if show_gestures:
        mode_text += "Visual "
    
    cv2.putText(image, f'Modes: {mode_text}', (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    screen_pos = f'Screen: {smooth_x if "result" in locals() and result.multi_hand_landmarks else "N/A"}, {smooth_y if "result" in locals() and result.multi_hand_landmarks else "N/A"}'
    cv2.putText(image, screen_pos, (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    cv2.putText(image, 'ESC:Exit M:Mirror G:Visual S:Scroll C:Click', (20, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imshow("Hand Tracking Control", image)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('m') or key == ord('M'):
        mirror_mode = not mirror_mode
    elif key == ord('g') or key == ord('G'):
        show_gestures = not show_gestures
    elif key == ord('s') or key == ord('S'):
        scroll_mode = not scroll_mode
    elif key == ord('c') or key == ord('C'):
        click_mode = not click_mode

cap.release()
cv2.destroyAllWindows()