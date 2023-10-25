import cv2
import numpy as np
from keras.models import load_model
from util import float_to_int
import time

def find_human_outline(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply a Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours in the edge image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a black canvas
    #outline_image = np.zeros_like(frame)
    
    # Draw the contours on the black canvas in white
    #cv2.drawContours(outline_image, contours, -1, (255, 255, 255), 2)

    ##############
    white_background = np.full_like(frame, (255, 255, 255))
    
    # Draw the contours on the white background in black
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline_image = cv2.drawContours(white_background, contours, -1, (0, 0, 0), 2)
    
    ##############
    
    return outline_image

if __name__ == '__main__':
    #print("Hello")

    model = load_model('./model_000500.h5', compile=False)

    # Initialize the webcam
    cap = cv2.VideoCapture(0)  # 0 represents the default camera (you can change it if you have multiple cameras)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        #print("Deb")
        ret, frame = cap.read()
        #print("SHAPE", frame.shape)
        dim = (512,512)
        t1 = time.time()
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        
        frame = (frame - 127.5) / 127.5
        frame = np.expand_dims(frame, axis=0)
        t2 = time.time()
        print("Preprocess Time:", t2-t1)
        #print("BAL Shapee", frame.shape)
        if not ret:
            break

        #outline_frame = find_human_outline(frame)
        t1 = time.time()
        gen_image = model.predict(frame)
        t2 = time.time()
        print("Time:", t2-t1)
        gen_image = (gen_image + 1) / 2.0
        gen_image=float_to_int(gen_image)
        gen_image = np.squeeze(gen_image)
        #print("GEN SHAPEXXXXXX", gen_image.shape)

        # Display the real-time outline
        cv2.imshow('Real-time Human Outline', gen_image)

        # Exit the loop by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
