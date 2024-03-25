import base64
import cv2
import numpy as np
import eventlet
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from mark_detector import MarkDetector
from pose_estimator import PoseEstimator

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
phone_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mobilephone.xml')

def readb64(uri):
    try:
        encoded_data = uri.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data + "==="), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        print(f"Error in readb64: {e}")
        return None

def detect_mobile_phone(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        phones = phone_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        return len(phones)
    except Exception as e:
        print(f"Error in detect_mobile_phone: {e}")
        return 0


@socketio.on('image')
def handle_image(data):
    try:
        image = readb64(data['img'])
        if image is None:
            emit('result', {"status": "error", "message": "Unable to read the image"})
            return

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        count = len(faces)
        if count == 0:
            response_message = "All clear"
        elif count == 1:
            response_message = "One person detected"
        else:
            response_message = f"{count} people detected"

        # Pose estimation
        pose_estimator = PoseEstimator(img_size=(image.shape[0], image.shape[1]))
        mark_detector = MarkDetector()
        frame = image
        facebox = mark_detector.extract_cnn_facebox(image)
        if facebox is not None:
            x1, y1, x2, y2 = facebox
            face_img = image[y1: y2, x1: x2]
            marks = mark_detector.detect_marks(face_img)
            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1
            pose = pose_estimator.solve_pose_by_68_points(marks)
            # print('pose1',pose)
            if pose is not None:
                img, pose = pose_estimator.draw_annotation_box(frame, pose[0], pose[1], color=(0, 255, 0))
                response_pose = pose
                print('pose->',pose)
            else:
                print('pose2', pose)
                response_pose = None

        else:
            print('pose3')
            response_pose = None

        _, img_encoded = cv2.imencode('.png', image)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        num_mobiles = detect_mobile_phone(image)
        print({"status": "success", "message": response_message, "num_people": count,"pose": response_pose,"num_mobiles": num_mobiles})
        emit('result', {"status": "success", "message": response_message, "num_people": count,
                        "image": img_base64, "pose": response_pose,"num_mobiles": num_mobiles})

    except Exception as e:
        emit('result', {"status": "error", "message": str(e)}, namespace='/')

if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 8080)), app)
