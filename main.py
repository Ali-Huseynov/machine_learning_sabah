import os
import face_recognition
from PIL import Image
import cv2
import numpy as np


# Function to load known faces from the dataset
def load_known_faces(people_folder):
    known_face_encodings = {}
    for person_name in os.listdir(people_folder):
        person_folder = os.path.join(people_folder, person_name)
        if os.path.isdir(person_folder):
            known_face_encodings[person_name] = []
            for filename in os.listdir(person_folder):
                image_path = os.path.join(person_folder, filename)
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)
                if len(face_encodings) > 0:
                    known_face_encodings[person_name].append(face_encodings[0])
    return known_face_encodings


# Function to recognize faces in the input image
def recognize_faces(image_path, known_face_encodings):
    image_name = image_path.split("/")[-1]
    original_image = cv2.cvtColor(np.array(Image.open(image_path)), cv2.COLOR_BGR2RGB)

    unknown_image = face_recognition.load_image_file(image_path)
    unknown_face_locations = face_recognition.face_locations(unknown_image)
    unknown_face_encodings = face_recognition.face_encodings(unknown_image, unknown_face_locations)

    face_locations = face_recognition.face_locations(unknown_image)

    for index, face_encoding in enumerate(unknown_face_encodings):
        name = "Unknown"
        for person_name, known_encodings in known_face_encodings.items():
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            if True in matches:
                name = person_name
                break

        (x, y, w, z) = face_locations[index]
        cv2.rectangle(original_image, (z - 10, w + 10), (y + 10, x - 10), (50, 205, 50), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(original_image, name, (z, x - 20), font, 0.7, (500, 500, 500), 2)

    os.makedirs("output", exist_ok=True)
    cv2.imwrite(f"output/{image_name}", original_image)


# Paths
people_folder = "PeopleData"  # Update this with your dataset path
input_images_path = "input/"  # Update this with your input image path

# Load known faces
known_face_encodings = load_known_faces(people_folder)

for input_image_name in os.listdir(input_images_path):
    # Recognize faces in the input image and save them in respective folders
    input_image_path = os.path.join(input_images_path, input_image_name)

    recognize_faces(input_image_path, known_face_encodings)

    print(input_image_path, " - Done!")
