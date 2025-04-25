import cv2
import imutils
import numpy as np
from centroidtracker import CentroidTracker
import yagmail
from tensorflow.keras.models import load_model
import time
from playsound import playsound
import pywhatkit as kit
import requests


autoencoder = load_model('autoencoder_model5.keras')
fire_cascade = cv2.CascadeClassifier('cascade.xml')
ppe_cascade = cv2.CascadeClassifier('cascade2.xml')

protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

tracker1 = CentroidTracker(maxDisappeared=80, maxDistance=90)
tracker2 = CentroidTracker(maxDisappeared=80, maxDistance=90)

MIN_SIZE_FOR_MOVEMENT = 1000
MOVEMENT_DETECTED_PERSISTENCE = 10
movement_persistent_counter = 400
people_count_camera2 = 0
last_mismatch_time = None
email_sent_flag = False
last_fire_alert_time = None
EMAIL_DELAY = 60
detection_timers = {}
last_email_time = 0
#phone_number = +4xxxxxxxxxxxxxx
#massage = "Upozornenie, počet osoôb sa nezhoduje!"
frame_count = 0

# Non-max suppression function
def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    if len(boxes) == 0 or boxes.ndim != 2 or boxes.shape[1] != 4:
        return np.array([])
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")



def check_protective_gear(frame, bbox):
    x1, y1, x2, y2 = bbox.astype(int)
    person_roi = frame[y1:y2, x1:x2]
    if person_roi is None or person_roi.size == 0:
        print("Chyba: Prázdny region záujmu.")
        return False
    gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
    detections = ppe_cascade.detectMultiScale(gray_roi, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(detections) > 0



def process_frame(cap, tracker, first_frame, delay_counter, is_camera_1, line_start, line_end, person_line_crossed):
    global movement_persistent_counter, people_count_camera2, last_email_time, massage, phone_number, frame_count
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Chyba: Nepodarilo sa načítať snímku z kamery.")
        return None, 0, first_frame, delay_counter, False
    frame = imutils.resize(frame, width=650, height=600)
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
    detector.setInput(blob)
    person_detections = detector.forward()



    rects = []

    for i in np.arange(0, person_detections.shape[2]):
        confidence = person_detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(person_detections[0, 0, i, 1])
            if CLASSES[idx] != "person":
                continue
            person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            rects.append(person_box)

    if len(rects) > 0:
        boundingboxes = np.array(rects).astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.3)
    else:
        rects = np.array([])
    boundingboxes = np.array(rects).astype(int)

    rects = non_max_suppression_fast(boundingboxes, 0.3)
    objects = tracker.update(rects)

    #if len(objects) < 2:
    #        movement_persistent_counter = 400

    for (objectId, bbox) in objects.items():
        x1, y1, x2, y2 = bbox.astype(int)
        has_ppe = check_protective_gear(frame, bbox)
        color = (0, 255, 0) if has_ppe else (0, 0, 255)
        text2 = f"ID: {objectId} {'(OK)' if has_ppe else '(Chyba ochrana!)'}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        text = f"ID osoby: {objectId}"
        cv2.putText(frame, text, (x1, y1 - 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text2, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 1)
        current_time = time.time()
        if not has_ppe:
            if objectId not in detection_timers:
                detection_timers[objectId] = current_time
            elif current_time - detection_timers[objectId] >= EMAIL_DELAY:
                yag = yagmail.SMTP("umb.upozornenie@gmail.com", "xxxxxxxxxxxxxx")
                yag.send("simox058@gmail.com", "Upozornenie: Chýba ochrana!",
                         f"Osoba s ID {objectId} nemá ochranné pomôcky!")
        else:
            if objectId in detection_timers:
                detection_timers.pop(objectId)

    if is_camera_1:
        frame_count += 1
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame_resized = cv2.resize(gray_frame, (64, 64))  # Resize to match input size of autoencoder
        gray_frame_resized = np.expand_dims(gray_frame_resized, axis=-1) / 255.0  # Normalize
        reconstructed_frame = autoencoder.predict(np.expand_dims(gray_frame_resized, axis=0))
        error = np.mean(np.abs(reconstructed_frame - gray_frame_resized))
        anomaly_threshold = 0.05
        if error > anomaly_threshold and frame_count % 100 == 0:
            screenshot_path = "anomaly_screenshot.jpg"
            cv2.imwrite(screenshot_path, frame)
            with open(screenshot_path, 'rb') as f:
                files = {'file': (screenshot_path, f)}
                response = requests.post("http://localhost:5000/upload", files=files)

            if response.status_code == 200:
                print("Anomália bola úspešne nahraná na server.")
            else:
                print("Chyba pri nahrávaní anomálie na server.")
            email_body = f"""
                        <html>
<body style="font-family: Arial, sans-serif;">
    <p style="font-size: 16px; color: #333; text-align: center;">Bola detegovaná anomália s chybou {error}. Je toto falošný poplach?</p>
    <div style="text-align: center; background-color: #f0f0f0; padding: 20px;">
        <a href="http://localhost:5000/response?answer=ANO&filename=anomaly_screenshot.jpg" style="background-color: #f44336; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin: 5px; display: inline-block; min-width: 80px; text-align: center; height: 36px; line-height: 16px;">
            ÁNO
        </a>
        <a href="http://localhost:5000/response?answer=NIE&filename=anomaly_screenshot.jpg" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin: 5px; display: inline-block; min-width: 80px; text-align: center; height: 36px; line-height: 16px;">
            NIE
        </a>
    </div>
</body>
</html>
                        """
            yag = yagmail.SMTP("umb.upozornenie@gmail.com", "xxxxxxxxxxxxxxxxxxx")
            yag.send("simox058@gmail.com", "Upozornenie", email_body, "anomaly_screenshot.jpg")





    transient_movement_flag = False
    if is_camera_1:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if first_frame is None:
            first_frame = gray

        delay_counter += 1
        if delay_counter > 10:
            delay_counter = 0
            first_frame = gray

        frame_delta = cv2.absdiff(first_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            if cv2.contourArea(c) > MIN_SIZE_FOR_MOVEMENT:
                transient_movement_flag = True
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if transient_movement_flag:
            movement_persistent_counter = 400
        else:
            if movement_persistent_counter > 0:
                movement_persistent_counter -= 1

        text = "Pocitadlo pohybu: " + str(movement_persistent_counter)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)




        if movement_persistent_counter == 250:
            playsound('warning.mp3', False)
        if movement_persistent_counter == 130:
            playsound('warning.mp3', False)
        if movement_persistent_counter == 10:
            yag = yagmail.SMTP("umb.upozornenie@gmail.com", "xxxxxxxxxxxxxxxxx")
            yag.send("simox058@gmail.com", "Upozornenie", "V laboratóriu sa nehýbe osoba")

        global last_fire_alert_time
        fire = fire_cascade.detectMultiScale(frame, 20, 12)
        if len(fire) > 0:
            current_time = time.time()
            if last_fire_alert_time is None or (current_time - last_fire_alert_time >= 15):
                for (x, y, w, h) in fire:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                yag = yagmail.SMTP("umb.upozornenie@gmail.com", "xxxxxxxxxxxxxxxxxx")
                yag.send("simox058@gmail.com", "Upozornenie", "V laboratóriu horí osoba")
                last_fire_alert_time = current_time
        else:
            last_fire_alert_time = None

    if not is_camera_1:
        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox.astype(int)
            line_crossed = False
            if objectId not in person_line_crossed:
                person_line_crossed[objectId] = {"entered": False, "exited": False}
            if not person_line_crossed[objectId]["entered"]:
                if y1 < line_start[1] < y2 and x1 < line_start[0] and x2 > line_end[0]:
                    people_count_camera2 += 1
                    person_line_crossed[objectId]["entered"] = True
                    line_crossed = True
            elif not person_line_crossed[objectId]["exited"]:
                if y1 > line_start[1] > y2 and x1 < line_start[0] and x2 > line_end[0]:
                    people_count_camera2 -= 1
                    person_line_crossed[objectId]["exited"] = True
                    line_crossed = True
            if line_crossed:
                person_line_crossed[objectId]["entered"] = True if y1 < line_start[1] else \
                person_line_crossed[objectId]["entered"]
                person_line_crossed[objectId]["exited"] = True if y1 > line_start[1] else person_line_crossed[objectId][
                    "exited"]
        cv2.line(frame, line_start, line_end, (255, 0, 0), 2)
    return frame, len(objects), first_frame, delay_counter, movement_persistent_counter



def main():
    global last_mismatch_time, email_sent_flag
    cap1 = cv2.VideoCapture("test2.mp4")
    cap2 = cv2.VideoCapture("test2.mp4")
    first_frame1 = None
    first_frame2 = None
    delay_counter1 = 0
    delay_counter2 = 0
    line_start = (0, 250)
    line_end = (600, 250)
    person_line_crossed = {}
    while True:
        frame1, count1, first_frame1, delay_counter1, movement_persistent_counter = process_frame(cap1, tracker1,
                                                                                                  first_frame1,
                                                                                                  delay_counter1,
                                                                                                  is_camera_1=True,
                                                                                                  line_start=None,
                                                                                                  line_end=None,
                                                                                                  person_line_crossed=person_line_crossed)
        frame2, people_count_camera2, first_frame2, delay_counter2, _ = process_frame(cap2, tracker2, first_frame2,
                                                                                      delay_counter2, is_camera_1=False,
                                                                                      line_start=line_start,
                                                                                      line_end=line_end,
                                                                                      person_line_crossed=person_line_crossed)


        cv2.putText(frame1, f"Pocet osob: {count1}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame2, f"Pocet osob: {people_count_camera2}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                    2)

        current_time = time.time()
        if count1 != people_count_camera2:
            if last_mismatch_time is None:
                last_mismatch_time = current_time
            if current_time - last_mismatch_time >= 60:
                #kit.sendwhatmsg(phone_number,massage)
                yag = yagmail.SMTP("umb.upozornenie@gmail.com", "xxxxxxxxxxxxxxxx")
                yag.send("simox058@gmail.com", "Upozornenie", "Počet osôb sa nezhoduje!")
                last_mismatch_time = current_time
        else:
            last_mismatch_time = None
        combined_frame = np.hstack((frame1, frame2))
        cv2.imshow("Kamera", combined_frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
