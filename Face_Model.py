import os
import face_recognition
import cv2
import numpy as np
import glob


class FaceModel:
    def __init__(self) -> None:

        self.face_embeddings = []
        self.face_names = []

        # resizing width and height
        self.frame_resizing = 0.25

    def load_images(self, image_path):

        # load images
        images_path = glob.glob(os.path.join(image_path, "*.*"))

        print("{} images found successfully".format(len(images_path)))

        # encode images and store the images and names

        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # get the filename

            base_name = os.path.basename(img_path)

            (filename, ext) = os.path.splitext(base_name)

            # img encoding

            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            # store the img name and encoding

            self.face_embeddings.append(img_encoding)

            self.face_names.append(filename)

            print("{} images encoded successfully")

    def detect_faces(self, frame):

        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Find all the faces and face encodings in the current frame of video
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.face_embeddings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = self.face_names[first_match_index]

            # # Or instead, use the known face with the smallest distance to the new face
            # face_distances = face_recognition.face_distance(self.face_embeddings, face_encoding)
            # best_match_index = np.argmin(face_distances)
            # if matches[best_match_index]:
            #     name = self.face_names[best_match_index]
            face_names.append(name)

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names
