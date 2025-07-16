import cv2

def capture_image(save_path):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(save_path, frame)
        return True
    else:
        return False
