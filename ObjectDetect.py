from tensorflow.keras.applications.resnet import preprocess_input
from imutils.object_detection import non_max_suppression
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.preprocessing.image import img_to_array
from helper import sliding_window
from helper import pyramid
import numpy as np
import imutils
import time
import cv2

WIDTH = 600
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = (300, 300)
INPUT_SIZE = (224, 224)

model = ResNet101(input_shape=(224, 224, 3), weights='imagenet', include_top=True)

orig = cv2.imread("./testImgs/hummingbird.jpg")
orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
orig = imutils.resize(orig, width=WIDTH)
(H, W) = orig.shape[:2]

pyram = pyramid(orig, PYR_SCALE, min_size=ROI_SIZE)

rois = []
locs = []

start = time.time()

for image in pyram:
    scale = W / float(image.shape[1])

    for (x, y, roi_orig) in sliding_window(image, WIN_STEP, ROI_SIZE):
        x = int(x * scale)
        y = int(y * scale)
        w = int(ROI_SIZE[0] * scale)
        h = int(ROI_SIZE[1] * scale)
        roi = cv2.resize(roi_orig, INPUT_SIZE)
        roi = img_to_array(roi)
        roi = preprocess_input(roi)

        rois.append(roi)
        locs.append((x, y, x + w, y + h))

end = time.time()
print("[INFO] looping over pyramid/windows took {:.5f} seconds".format(end - start))

rois = np.array(rois, dtype='float32')
print("[INFO] classifying ROIs...")
start = time.time()
preds = model.predict(rois)
end = time.time()
print("[INFO] classifying ROIs took {:.5f} seconds".format(end - start))

preds = imagenet_utils.decode_predictions(preds, top=1)
labels = {}

for (i, p) in enumerate(preds):
    (imagenetID, label, prob) = p[0]
    if prob >= 0.6:
        box = locs[i]
        L = labels.get(label, [])
        L.append((box, prob))
        labels[label] = L

for label in labels.keys():
    print("[INFO] showing results for '{}'".format(label))
    clone = orig.copy()

    for (box, prob) in labels[label]:
        (startX, startY, endX, endY) = box
        cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)

    cv2.imshow("Before", clone)
    clone = orig.copy()
    clone = orig.copy()

boxes = np.array([p[0] for p in labels[label]])
proba = np.array([p[1] for p in labels[label]])
boxes = non_max_suppression(boxes, proba)

for (startX, startY, endX, endY) in boxes:
    cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(clone, f"{label}-{prob}", (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)


cv2.imshow("After", clone)
cv2.waitKey(0)
print(label)
print(clone.shape)
