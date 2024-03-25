import base64
import matplotlib.pyplot as plt
from flask import Flask

import os
from flask import request, jsonify
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from flask_cors import CORS

from mark_detector import MarkDetector
from pose_estimator import PoseEstimator

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# multiple_people_detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/d0/1")



# @app.route('/predict_people',methods=['GET','POST'])
# def predict_pose():
#     data = request.get_json()
#     # get image tensor
#     output = multiple_people_detector(image, threshold = 0.5)
#     people = 0
#     for i in range(int(output['num_detections'][0])):
#         if classes[i] == 1 and scores[i] > threshold:
#             people += 1
#             ymin, xmin, ymax, xmax = boxes[i]
#             (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
#                                           ymin * im_height, ymax * im_height)
#             draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
#                       width=4, fill='red')

#     return jsonify({ 'people' : int(people) , 'image' : image})


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


#
# def readb64(uri):
#    encoded_data = uri.split(',')[1]
#    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
#    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    return img

# def readb64(uri):
#     try:
#         encoded_data = uri.split(',')[1]
#         nparr = np.frombuffer(base64.b64decode(encoded_data + "==="), np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         return img
#     except Exception as e:
#         print(f"Error in readb64: {e}")
#         return None

def readb64(uri):
    try:
        encoded_data = uri.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data + "==="), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:  # Check if the image was decoded successfully
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        print(f"Error in readb64: {e}")
        return None



@app.route('/predict_people', methods=['POST'])
def predict_people():
    try:
        # Get the base64-encoded image from the request
        data = request.get_json(force=True)
        image = r'{}'.format(data['img'])

        # Read the base64-encoded image using the readb64 function
        image = readb64(image)

        # Check if the image is loaded successfully
        if image is None:
            return jsonify({"status": "error", "message": "Unable to read the image"}), 400

        # Convert the image to grayscale for better performance
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Draw rectangles around the detected faces (heads)
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get the count of detected faces (heads)
        count = len(faces)

        # Create the response message based on the count
        if count == 0:
            response_message = "All clear"
        elif count == 1:
            response_message = "One person detected"
        else:
            response_message = f"{count} people detected"

        # Convert the image to base64 for response
        _, img_encoded = cv2.imencode('.png', image)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        return jsonify({"status": "success", "message": response_message, "num_people": count})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/predict_pose', methods = ['GET', 'POST'])
def predict_pose() :
    data = request.get_json(force = True)
    image = r'{}'.format(data['img'])
    # print(type(image), image)
    image= readb64(image)
    # plt.imshow(image)
    # plt.show()
    # plt.imsave(image, 'sample.jpg');
    height, width = image.shape[0], image.shape[1]
    pose_estimator = PoseEstimator(img_size=(height, width))
    mark_detector = MarkDetector()

    facebox = mark_detector.extract_cnn_facebox(image)
        # Any face found?
    frame = image
    if facebox is not None:

        # Step 2: Detect landmarks. Crop and feed the face area into the
        # mark detector.
        x1, y1, x2, y2 = facebox
        face_img = frame[y1: y2, x1: x2]

        # Run the detection.
        marks = mark_detector.detect_marks(face_img)

        # Convert the locations from local face area to the global image.
        marks *= (x2 - x1)
        marks[:, 0] += x1
        marks[:, 1] += y1

        # Try pose estimation with 68 points.
        pose = pose_estimator.solve_pose_by_68_points(marks)

        # All done. The best way to show the result would be drawing the
        # pose on the frame in realtime.

        # Do you want to see the pose annotation?
        img, pose = pose_estimator.draw_annotation_box(frame, pose[0], pose[1], color=(0, 255, 0))

        # Do you want to see the head axes?
        # pose_estimator.draw_axes(frame, pose[0], pose[1])

        # Do you want to see the marks?
        # mark_detector.draw_marks(frame, marks, color=(0, 255, 0))

        # Do you want to see the facebox?
        # mark_detector.draw_box(frame, [facebox])
        img = list(img)
        print("response to server -> ",pose)
        return jsonify({'img' : 'face found', 'pose' : pose})
    else :
        return jsonify({'message' : 'face not found', 'img' : 'img'})




# @app.route('/predict_people',methods=['GET','POST'])
# def predict() :
#     data = request.get_json(force = True)
#     image= readb64(data['img'])
#     im_width, im_height = image.shape[0], image.shape[1]
#     image = image.reshape((1, image.shape[0], image.shape[1], 3))
#     # return jsonify({'data' : data})
#     data = multiple_people_detector(image)
#
#     boxes = data['detection_boxes'].numpy()[0]
#     classes = data['detection_classes'].numpy()[0]
#     scores = data['detection_scores'].numpy()[0]
#
#     threshold = 0.5
#     people = 0
#     for i in range(int(data['num_detections'][0])):
#         if classes[i] == 1 and scores[i] > threshold:
#             people += 1
#             ymin, xmin, ymax, xmax = boxes[i]
#             (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
#                                           ymin * im_height, ymax * im_height)
#             # draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
#             #           width=4, fill='red')
#
#     return jsonify({ 'people' : int(people) , 'image' : 'image'})


@app.route('/save_img', methods=['GET', 'POST'])
def save() :
    data = request.get_json(force = True)
    image = r'{}'.format(data['img'])
    user = data['user']
    image= readb64(image)
    base_dir = os.getcwd()
    path = r"{}\images\{}.jpg".format(base_dir, user[0:-10])
    print(path)
    plt.imsave(image, path)
    return jsonify({'path' : path})


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0',port=8080)