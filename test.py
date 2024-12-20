import cv2
import numpy as np
import os

# Data
dataset_path = "./data/"
facedata = []
labels = []
classId = 0
namemap = {}

for f in os.listdir(dataset_path):
    if f.endswith(".npy"):
        namemap[classId] = f[:-4]
        # x values
        dataItem = np.load(os.path.join(dataset_path, f))
        m = dataItem.shape[0]
        facedata.append(dataItem)

        # y values
        target = classId * np.ones((m,))
        classId += 1
        labels.append(target)

xt = np.concatenate(facedata, axis=0)
yt = np.concatenate(labels, axis=0).reshape((-1, 1))
print(xt.shape)
print(yt.shape)
print(namemap)

# Algorithm
def dist(p, q):
    return np.sqrt(np.sum((p - q) ** 2))

def knn(x, y, query_point, k=5):
    m = x.shape[0]
    dlist = []

    for i in range(m):
        d = dist(x[i], query_point)
        dlist.append((d, y[i]))

    dlist = sorted(dlist, key=lambda x: x[0])

    # Print debugging information
    print("Distances and labels:", dlist[:k])

    dlist = np.array(dlist[:k], dtype=object)  # Ensure homogeneity
    
    labels = [label for (_, label) in dlist]
    labels, cnts = np.unique(labels, return_counts=True)
    idx = cnts.argmax()
    pred = labels[idx]

    return int(pred)

# Prediction
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Camera not successfully opened...")
    exit()

model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# List to save images
skip = 0
offset = 20
while True:
    success, img = cam.read()
    if not success:
        print('Capture failed due to error!')
        break

    gryimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = model.detectMultiScale(gryimg, 1.3, 5)
    
    for f in faces:
        x, y, w, h = f
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 34, 221), 2)

        cropped_face = img[y - offset:y + h + offset, x - offset:x + w + offset]
        cropped_face = cv2.resize(cropped_face, (100, 100))

        # Add debug print for shape and type of cropped_face
        print("Cropped face shape:", cropped_face.flatten().shape)
        print("Cropped face dtype:", cropped_face.flatten().dtype)
        
        classPredicted = knn(xt, yt, cropped_face.flatten())
        namepredicted = namemap[classPredicted]
        cv2.putText(img, namepredicted, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 145, 220), 2, cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Predicted Window", img)
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
