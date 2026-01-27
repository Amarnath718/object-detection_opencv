import cv2
import numpy as np
from google.colab.patches import cv2_imshow

#load image

image=cv2.imread("/content/dog.jpg")
height,width=image.shape[:2]

scale=0.00392

#load class names
with open("/content/yolov3.txt",'r') as f:
  classes=[line.strip()for line in f.readlines()]

#generate random colors for each class
COLORS=np.random.uniform(0,255,size=(len(classes),3))

#load yolo model

#Load YOLO model
net=cv2.dnn.readNet(
    "/content/yolov3.weights",
    "/content/yolov3.cfg"
)

#Create input blob
blob=cv2.dnn.blobFromImage(
    image,
    scale,
    (416,416),
    (0,0,0),
    swapRB=True,
    crop=False
)
net.setInput(blob)

#get output layer names

def get_output_layers(net):
  layer_names = net.getLayerNames()
  return [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

#draw bounding box

def draw_bounding_box(img,class_id,confidence,x,y,x_plus_w,y_plus_h):
  label=f"{classes[class_id]}:{confidence:.2f}"
  color=COLORS[class_id]

  cv2.rectangle(img,(x,y),(x_plus_w, y_plus_h), color, 2)
  cv2.putText(
      img,
      label,
      (x,y-10),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.5,
      color,
      2
)

#forward pass
outs = net.forward(get_output_layers(net))

#detection parameters

class_ids = []
confidences = []
boxes = []

conf_threshhold=0.5
nms_threshhold=0.4

#process detection

for out in outs:
  for detection in out:
    scores = detection[5:]
    class_id = np.argmax(scores)
    confidence = scores[class_id]

    if confidence>conf_threshhold:
      center_x=int(detection[0]*width)
      center_y=int(detection[1]*height)
      w = int(detection[2]*width)
      h = int(detection[3]*height)

      x = int(center_x-w/2)
      y = int(center_y-h/2)

      class_ids.append(class_id)
      confidences.append(float(confidence))
      boxes.append([x,y,w,h])

#apply non-max suppression

indices = cv2.dnn.NMSBoxes(
    boxes,
    confidences,
    conf_threshhold,
    nms_threshhold
)

#draw final detection

if len(indices)>0:
  for i in indices.flatten():
    x,y,w,h =boxes[i]

    draw_bounding_box(
        image,
        class_ids[i],
        confidences[i],
        x,
        y,
        x+w,
        y+h
    )

#display and save result (colab)

cv2_imshow(image)
cv2.imwrite("/content/object-detection.jpg",image)
