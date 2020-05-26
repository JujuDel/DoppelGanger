# DoppelGanger tool

This project aims to find a doppelganger or look-alike to a given person within a celebrity dataset.

The repo after unzipping the `data.zip` should be organized as follow:
```
DoppelGanger
├── DoppelGanger.py
├── README.md
└── data
    ├── celeb_mapping.npy
    ├── encoding_celeb_mini.dat
    └── images
        └── celeb_mini
        │   ├── n00000001
        │   │   └── *.JPEG
        │   ├── n00000003
        │   │   └── *.JPEG
        │   ├── ...
        └── test-images
            └── *.jpg
```
The images and the `celeb_mapping.npy` are coming from [https://courses.opencv.org](https://courses.opencv.org/).

The steps performed are explained below.

## 1. Encode the known faces

The function `load_encodings` loads the given `encoding_celeb_mini.dat` which contains the encoding of the faces of the celebrities. If the files doesn't exist, it will create it

Below is a shorten extract of the code:
```python
# Compute the 128-dimension face encoding for each faces
def encode_faces(path=FACE_DATA_PATH):
    # Known face dataset
    faces_encodings = {}
    # All the folders inside the path
    allCelebs = [d for d in os.listdir(path)
                    if os.path.isdir(os.path.join(path, d))]
    # For each folder
    for i, celebID in enumerate(allCelebs):
        # Get all the images in the folder
        currPath = os.path.join(path, celebID)
        allImages = glob.glob(os.path.join(currPath, '*.JPEG'))
        #For each images
        for pathImg in allImages:
            # Load the image into a numpy array
            image = face_recognition.load_image_file(pathImg, 'RGB')
            # Compute the 128-dimension face encoding for each faces
            encodings = face_recognition.face_encodings(image)
            if len(encodings) == 1:
                # We keep only the images with exactly 1 face
                filename = pathImg.replace('\\', '/').split('/')[-1]
                celeb_ID = filename.split('_')[0]
                faces_encodings[(filename, celeb_ID)] = encodings[0]
    return faces_encodings
```

## 2. Compute and compare the similarities

Once we have the encodings of our dataset, the goal is to compute the similarities between the encoding of a given image and the known encodings.

The doppelganger or looka-like celebrity will then be the one with the more similarities.

Below is a shorten extract of the code:
```python
# Compute the 128-dimension face encoding for each faces
for imgPath in allImgPaths:
    # Read the test image
    im = cv2.imread(imgPath)
    # Compute the 128-dimension face encoding for each faces
    image = face_recognition.load_image_file(pathImg, 'RGB')
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
            # Here is the doppelganger!
            (filename, celeb_ID) = face_names[best_match_idx]
```

## 3. Results

Below are the computed doppelganger with the given input images.

Input image | DoppelGanger
:---:|:---:
shashikant-pedwal.jpg | Amitabh Bachchan
<img height="256" alt="shashikant-pedwal.jpg" src="https://github.com/JujuDel/DoppelGanger/blob/master/examples/shashikant-pedwal.jpg"> | ![](https://github.com/JujuDel/DoppelGanger/blob/master/examples/n00000102_00000547.JPEG)
sofia-solares.jpg | Selena
<img height="256" alt="sofia-solares.jpg" src="https://github.com/JujuDel/DoppelGanger/blob/master/examples/sofia-solares.jpg"> | ![](https://github.com/JujuDel/DoppelGanger/blob/master/examples/n00002238_00000655.JPEG)
me.jpg | Jake Johnson
<img height="256" alt="me.jpg" src="https://github.com/JujuDel/DoppelGanger/blob/master/examples/me.jpg"> | ![](https://github.com/JujuDel/DoppelGanger/blob/master/examples/n00000920_00000386.JPEG)