from flask import Flask, render_template, Response, request, jsonify
from imutils.video import VideoStream
import cv2
from flask_cors import CORS
import numpy as np
import time
import base64
import zlib
import facenet
import align.detect_face
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import imutils
import pickle

import threading
# Create a lock for synchronization
frame_lock = threading.Lock()

app = Flask(__name__)
CORS(app)

MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
INPUT_IMAGE_SIZE = 160
CLASS_NAMES = {
    0: "Han Van Luan;B20DCCN410",
    1: "Tran Dinh Tinh;B20DCCN620",
    2: "Nguyen Quoc Anh;B20DCCN026",
    3: "Ngo Cong Son;B20DCCN567",
    5: "Đo Huyen Trang;B20DCCN686",
}
CLASSIFIER_PATH = 'Models/facemodel.pkl'
FACENET_MODEL_PATH = 'Models/20180402-114759.pb'

@app.route('/')
def index():
    return render_template('index.html')

# Initialize the name variable as a global variable
name = ""
frame = None
def generate_frames():
    global name
    global frame
    name =""

    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    print("Custom Classifier, Successfully loaded")

    with tf.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():
            facenet.load_model(FACENET_MODEL_PATH)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")

            # IP camera stream
            # ip_camera_url
            # cap = cv2.VideoCapture(ip_camera_url)
            cap = VideoStream(src=0).start()

            while True:
                frame = cap.read()
                frame = imutils.resize(frame, width=600)
                frame = cv2.flip(frame, 1)

                bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

                faces_found = bounding_boxes.shape[0]

                try:
                    if faces_found > 0:
                        det = bounding_boxes[:, 0:4]
                        bb = np.zeros((faces_found, 4), dtype=np.int32)

                        for i in range(faces_found):
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]

                            if (bb[i][3] - bb[i][1]) / frame.shape[0] > 0.25:
                                cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                                scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                                scaled = facenet.prewhiten(scaled)
                                scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                                emb_array = sess.run(embeddings, feed_dict=feed_dict)

                                predictions = model.predict_proba(emb_array)
                                best_class_indices = np.argmax(predictions, axis=1)
                                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                                # best_name = class_names[best_class_indices[0]]

                                if best_class_probabilities > 0.7:
                                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                    text_x = bb[i][0]
                                    text_y = bb[i][3] + 20

                                    name = class_names[best_class_indices[0]]
                                    cv2.putText(frame, CLASS_NAMES[int(name)], (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 255, 255), thickness=1, lineType=2)
                                    cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17),
                                                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 255, 255), thickness=1, lineType=2)
                                else:
                                    name = "Unknown"
                                    cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 255, 255), thickness=1, lineType=2)

                except Exception as e:
                    print(e)

                ret, jpeg = cv2.imencode('.jpg', frame)
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
                


    cap.release()


def encode_and_compress_image(img):
    # Convert the frame to JPEG format
    _, jpeg = cv2.imencode('.jpg', img)
    frame_bytes = jpeg.tobytes()

    # Base64 encode and compress the image
    compressed_image = base64.b64encode(frame_bytes).decode('utf-8')
    return compressed_image

# def encode_and_compress_image(img):
#     # Convert the frame to JPEG format
#     _, jpeg = cv2.imencode('.jpg', img)
#     frame_bytes = jpeg.tobytes()

#     # Compress the image data using zlib
#     compressed_data = zlib.compress(frame_bytes)

#     # Base64 encode the compressed data
#     compressed_image = base64.b64encode(compressed_data).decode('utf-8')

#     return compressed_image


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/receive', methods=['POST'])
def receive_data():
    global name
    global frame
    data_from_java = request.data.decode('utf-8')
    start_time = time.time()
    elapsed_time = 0
    print(data_from_java)
    while elapsed_time < 5:
        if name == "Unknown" or name is None or not name: 
            time.sleep(1)  # Sleep for 1 second
            elapsed_time = time.time() - start_time
            
        elif CLASS_NAMES[int(name)].split(";")[1] == data_from_java:
            print(name)  
            name = "" 
            response_data = {'code': data_from_java, 'img': encode_and_compress_image(frame), 'message': True}
            return jsonify(response_data)
        else:
            time.sleep(1)  # Sleep for 1 second
            elapsed_time = time.time() - start_time

    name = "" 
    response_data = {'code': data_from_java, 'img': None, 'message': False}
    return jsonify(response_data)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6868)
