import cv2
import numpy as np
import os
import imutils
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

# Allow GPU memory growth for TensorFlow
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Load YOLO model
def load_yolo_model(weights_path, config_path):
    net = cv2.dnn.readNet(weights_path, config_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return net

# Load helmet classification model
def load_classification_model(model_path):
    return load_model(model_path)

# Prepare YOLO output layers
def get_yolo_output_layers(net):
    layer_names = net.getLayerNames()
    return [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Predict helmet status
def predict_helmet_status(model, helmet_roi):
    try:
        helmet_roi = cv2.resize(helmet_roi, (224, 224)).astype('float32') / 255.0
        helmet_roi = helmet_roi.reshape(1, 224, 224, 3)
        return int(model.predict(helmet_roi)[0][0])
    except Exception as e:
        print(f"Error in helmet detection: {e}")
        return None

# Draw detections and classify helmet status
def draw_detections(img, boxes, confidences, class_ids, model):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    COLORS = [(0, 255, 0), (0, 0, 255)]  # Green for bikes, Red for number plates

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            color = [int(c) for c in COLORS[class_ids[i]]]
            if class_ids[i] == 0:  # Bike
                helmet_roi = img[max(0, y):max(0, y) + max(0, h) // 4, max(0, x):max(0, x) + max(0, w)]
            else:  # Number plate
                x_h, y_h = x - 60, y - 350
                w_h, h_h = w + 100, h + 100
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 7)

                if y_h > 0 and x_h > 0:
                    h_r = img[y_h:y_h + h_h, x_h:x_h + w_h]
                    c = predict_helmet_status(model, h_r)
                    if c is not None:
                        cv2.putText(img, ['helmet', 'no-helmet'][c], (x, y - 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                        cv2.rectangle(img, (x_h, y_h), (x_h + w_h, y_h + h_h), (255, 0, 0), 10)

# Process video frames
def process_video(video_path, yolo_net, helmet_model):
    cap = cv2.VideoCapture(video_path)
    output_layers = get_yolo_output_layers(yolo_net)

    while True:
        ret, img = cap.read()
        if not ret:
            break

        img = imutils.resize(img, height=500)
        boxes, confidences, class_ids = process_frame(img, yolo_net, output_layers, helmet_model)
        draw_detections(img, boxes, confidences, class_ids, helmet_model)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) == 27:  # Escape key
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to handle video file selection and processing
def select_video():
    video_path = filedialog.askopenfilename(title="Select a Video", filetypes=[("Video Files", "*.mp4;*.avi")])
    if video_path:
        try:
            process_video(video_path, yolo_net, helmet_model)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

# Initialize GUI
def initialize_gui():
    root = tk.Tk()
    root.title("Helmet Detection Video Processor")

    select_button = tk.Button(root, text="Select Video", command=select_video)
    select_button.pack(pady=20)

    root.geometry("300x100")
    root.mainloop()

# Load models
yolo_net = load_yolo_model("yolov3-custom_7000.weights", "yolov3-custom.cfg")
helmet_model = load_classification_model('helmet-nonhelmet_cnn.h5')
print('Models loaded!')

# Run the GUI
if __name__ == "__main__":
    initialize_gui()
