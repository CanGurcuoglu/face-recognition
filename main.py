import os
import sys
import face_recognition
import cv2
import numpy as np
import math

def get_confidence(distance, threshold=0.6):
    range_ = 1.0 - threshold
    linear_value = (1.0 - distance) / (range_ * 2.0)

    if distance > threshold:
        return f"{round(linear_value * 100)}% confidence"
    else:
        value = (linear_value + ((1.0 - linear_value) * (math.pow(linear_value - 0.5, 2) * 2))) * 100
        return f"{round(value)}% confidence"

class FaceRecognition:
    def __init__(self):
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.known_face_encodings = []
        self.known_face_names = []
        self.process_this_frame = True
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir("faces"):
            face_img = face_recognition.load_image_file(f"faces/{image}")
            face_encoding = face_recognition.face_encodings(face_img)[0] if face_recognition.face_encodings(face_img) else None

            if face_encoding is not None:
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(os.path.splitext(image)[0])
        print("Encoded faces:", self.known_face_names)

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            print("Error: Could not open video device")
            sys.exit()

        while True:
            ret, frame = video_capture.read()

            if self.process_this_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                #rgb_small_frame = small_frame[:, :, ::-1]
                rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = "Unknown"

                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = get_confidence(face_distances[best_match_index])
                    self.face_names.append(f"{name} ({confidence})")

            self.process_this_frame = not self.process_this_frame

            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    faceReg = FaceRecognition()
    faceReg.run_recognition()
