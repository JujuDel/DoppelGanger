# -*- coding: utf-8 -*-

import os
import glob
import cv2
import numpy as np
import face_recognition
import pickle


###############################################################################
#
#    GLOBAL VARIABLES
#
###############################################################################

# Path to the folder of all the images
FACE_DATA_PATH = 'data/images/celeb_mini'

# Path to the folder of the images to test
FACE_TEST_PATH = 'data/images/test-images'

# Path of the data face encodings
FACE_ENCODINGS_PATH = 'data/encoding_celeb_mini.dat'

# Label -> Name Mapping file
labelMap = np.load("data/celeb_mapping.npy", allow_pickle=True).item()


###############################################################################
#
#    FACE ENCODING
#
###############################################################################

# Compute the 128-dimension face encoding for each faces
def encode_faces(path=FACE_DATA_PATH):
    # Known face dataset
    faces_encodings = {}

    # All the folders inside the path
    allCelebs = [d for d in os.listdir(path)
                    if os.path.isdir(os.path.join(path, d))]

    # Count the total amount of images
    # Fast and use only for printing values
    nbImages = 0
    for celebID in allCelebs:
        currPath = os.path.join(path, celebID)
        nbImages += len( glob.glob(os.path.join(currPath, '*.JPEG')) )
    print("{} Images".format(nbImages))

    # Counters
    nbDone = nbImageSkiped = 0

    # For each folder
    for i, celebID in enumerate(allCelebs):
        # Print the current avancement
        print("{} / {} - {} % - Encoding {} faces...".format(
               i, len(allCelebs),
               round(100. * nbDone / nbImages, 2),
               labelMap[celebID]))

        # Get all the images in the folder
        currPath = os.path.join(path, celebID)
        allImages = glob.glob(os.path.join(currPath, '*.JPEG'))

        #For each images
        for pathImg in allImages:
            nbDone += 1
            # Load the image into a numpy array
            image = face_recognition.load_image_file(pathImg, 'RGB')
            # Compute the 128-dimension face encoding for each faces
            encodings = face_recognition.face_encodings(image)
            if len(encodings) == 1:
                # We keep only the images with exactly 1 face
                filename = pathImg.replace('\\', '/').split('/')[-1]
                celeb_ID = filename.split('_')[0]
                faces_encodings[(filename, celeb_ID)] = encodings[0]
            else:
                nbImageSkiped += 1

    return faces_encodings, nbImages, nbImageSkiped


# Load a given `.dat` file or compute it if it doesn't exists
def load_encodings(path=FACE_ENCODINGS_PATH):
    encodings = {}
    if os.path.exists(path):
        # Load the file
        with open(path, 'rb') as f:
            encodings = pickle.load(f)
    else:
        # The file doesn't exist, encodes the default faces and dump the
        # encodings for a futur use
        print("The file {} doesn't exist.".format(path))
        print("Creation of default encodings from the given dataset...")
        encodings, nbDone, nbSkipped = encode_faces()

        print("\nEncoding finished!")
        print("  {} images used but {} skipped".format(nbDone, nbSkipped))
        print("\nDumping the encoded faces")
        dump_encodings(encodings, path)
    return encodings


# Dump the encodings
def dump_encodings(encodings, path=FACE_ENCODINGS_PATH):
    with open(path, 'wb') as f:
        pickle.dump(encodings, f)


###############################################################################
#
#    MAIN
#
###############################################################################

def main():
    # The images to test
    testImages = glob.glob(os.path.join(FACE_TEST_PATH, '*.jpg'))

    # Loads the encodings of the data to compare the test with
    encodings = load_encodings()
    face_names = list(encodings.keys())
    face_encodings = np.array(list(encodings.values()))

    print("Results:")

    # For each test image
    for test in testImages:
        celeb_name = "Unknown"
        celeb_img_path = ""

        # Read the test image
        im = cv2.imread(test)

         # Compute the 128-dimension face encoding for each faces
        image = face_recognition.load_image_file(test, 'RGB')
        test_encodings = face_recognition.face_encodings(image)

        # We keep only the images with exactly 1 face
        if len(test_encodings) == 1:
            test_encoding = test_encodings[0]

            # Compute the similarity with the known faces
            matches = face_recognition.compare_faces(
                face_encodings, test_encoding)

            # Compute the distance with the known faces
            face_distances = face_recognition.face_distance(
                face_encodings, test_encoding)
            best_match_idx = np.argmin(face_distances)

            # If the closest is a match
            if matches[best_match_idx]:
                celeb_info = face_names[best_match_idx]
                # Get the path to the celeb img
                celeb_img_path = os.path.join(
                        FACE_DATA_PATH, celeb_info[1], celeb_info[0])
                # Get the celeb name
                celeb_name = labelMap[celeb_info[1]]

            # Print and show the result
            print("{} - {}".format(celeb_name, celeb_img_path))
            cv2.imshow(
                    "Test: {}".format(test.replace('\\', '/').split('/')[-1]),
                    im)

            if celeb_img_path != "":
                cv2.imshow(
                        "Celeb Look-Alike = {}".format(celeb_name),
                        cv2.imread(celeb_img_path))

            # Wait for key input before continuing with the next image
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        else:
            print(
                "{} face(s) detected on the test. We need excatly 1!".format(
                        len(test_encodings)))
            print(
                "   Test {} skipped\n".format(
                        test.replace('\\', '/').split('/')[-1]))


if __name__ == '__main__':
    main()
