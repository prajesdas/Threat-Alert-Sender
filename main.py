import cv2
import torch
from ultralytics import YOLO
from utilis import YOLO_Detection, label_detection
import numpy as np
from send_email import send_email_with_image


def load_yolo_model(model_path):
    """Load the YOLO model and move it to the appropriate device."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = YOLO(model_path)
    model.to(device)
    model.nms = 0.7
    return model

def initialize_video_capture(video_source):
    """Initialize video capture from a camera or video file."""
    cap = cv2.VideoCapture(video_source)
    return cap

def display_frame(frame, dx=1024, dy=820, key = 10):
    """Display the current frame."""
    cv2.imshow("Frame", frame)
    if cv2.waitKey(key) & 0xFF == ord("q"):
        return True
    return False

def is_near_top_left(person_box, gun_box, threshold=100):
    """Check if the gun is near the top-left corner of the person box."""
    px1, py1, px2, py2 = person_box
    gx1, gy1, gx2, gy2 = gun_box

    # Calculate the top-left corner of the person box and the center of the gun box
    person_top_left = (px1, py1)
    gun_center = ((gx1 + gx2) / 2, (gy1 + gy2) / 2)

    # Calculate Euclidean distance
    distance = np.sqrt((person_top_left[0] - gun_center[0]) ** 2 + (person_top_left[1] - gun_center[1]) ** 2)
    return distance < threshold

def display_alert(frame, message):

    cv2.rectangle(frame, (10, 0), (200, 20), (0,69,255), -1)

    cv2.putText(frame, message, (15,15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245,245,245), 1, cv2.LINE_AA)


    return frame


def main():

    smtp_server = 'smtp.gmail.com'  # For Gmail
    port = 587  # For TLS
    sender_email = ''  # Replace with your email
    password = ''  # Replace with your email password or app password
    receiver_email = ''  # Replace with receiver's email
    subject = 'Robbery Alert'
    body = 'Please Find the attached footage of Robbery and Call Emergency.'


    # Load model
    model = load_yolo_model("robbery.pt")

    # Initialize video capture
    cap = initialize_video_capture("input_video/robbing.mp4")

    # Capture the original frame rate and video dimensions
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Original FPS: {fps}, Width: {width}, Height: {height}")
    gun_detected = 0
    cnt = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection
        boxes, classes, names, confidences, ids = YOLO_Detection(model, frame, conf=0.3)
        person_boxes = []
        gun_boxes = []

        # Separate detections for people and guns
        for box, cls, id in zip(boxes, classes, ids):
            x1, y1, x2, y2 = box
            name = names[int(cls)]
            if name == "person":
                person_boxes.append((box, id))  # store person boxes with their ID
            elif name == "gun":
                gun_boxes.append(box)  # store only the gun box

        # Check for proximity to top-left corner and draw boxes
        for (person_box, person_id) in person_boxes:
            gun_near_top_left = any(is_near_top_left(person_box, gun_box) for gun_box in gun_boxes)

            # Draw red box if gun is near top-left corner of the person box, else default color
            if gun_near_top_left:
                label_detection(frame=frame, text="Threat", tbox_color=(0, 0, 255), left=person_box[0], top=person_box[1],
                                bottom=person_box[2], right=person_box[3])  # Red box for alert
                gun_detected += 1
            else:
                label_detection(frame=frame, text="Person", tbox_color=(255, 144, 30), left=person_box[0], top=person_box[1],
                                bottom=person_box[2], right=person_box[3], )

        if gun_detected == 30:
            cv2.imwrite("robbery_image.png", frame)
            image_path = "robbery_image.png"  # Replace with the path to your image

            send_email_with_image(smtp_server, port, sender_email, password, receiver_email, subject, body, image_path)

        if gun_detected in range(30, 80):
            # if gun_detected > 30 :
            frame = display_alert(frame, "Email Sent to YAHYA")


        # Display the frame
        if display_frame(frame, key = 10):
            break


    # Release video capture resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
