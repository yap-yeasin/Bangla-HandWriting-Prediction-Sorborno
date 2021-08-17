import cv2
import numpy as np
import pandas as pd

net = cv2.dnn.readNet('yolov3_1200.weights','yolov3_custom.cfg')

# ## To enable Cuda
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


#save all the names in file o the list classes
classes = []
# with open("classes_N.names", "r") as f:
with open("classes.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    
#get layers of the network
layer_names = net.getLayerNames()

#Determine the output layer names from the YOLO model 
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

## Capture frame-by-frame 

img = cv2.imread("test/002.png")
# img = cv2.imread("draw/image.jpg")
# img = cv2.imread("test/101.bmp")
# img = cv2.imread("test/12.jpg")


img = cv2.resize(img, None, fx=0.6, fy=0.6)
# img = cv2.resize(img,None,fx=0.4,fy=0.4, interpolation = cv2.INTER_CUBIC)
height, width, channels = img.shape


# Using blob function of opencv to preprocess image
blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),swapRB=True, crop=False)
#Detecting objects
net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

#We use NMS function in opencv to perform Non-maximum Suppression
#we give it score threshold and nms threshold as arguments.
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
colors = np.random.uniform(0, 255, size=(len(classes), 3))
# print(boxes)
    

obj = []
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        # print(i)
        label = str(classes[class_ids[i]])
    
        obj.append([label,x,y,x + w, y + h ,(2*x+w)/2,(2*y+h)/2])
        
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y -5),cv2.FONT_HERSHEY_SIMPLEX,
			1/2, color, 2)

cv2.imshow("Object_Detect_By_Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows() # press space to destroy all window

Dt = {1: 'অ', 2: 'আ', 3:'ই',4:'ঈ',5:'উ',6:'ঊ',7:'ঋ',8:'এ',9:'ঐ',10:'ও',11:'ঔ'}
# Dt = {1: 'a', 2: 'b', 3:'c',4:'d',5:'e',6:'f',7:'g',8:'q',9:'w',10:'o',11:'l'}
# print(Dt.keys())
# print(Dt.values())
# print(label)

l = int(label) 
# print(type(label))

if l in Dt.keys():
    print(Dt[l])

# print('Object Detected ',len(obj))
# print(obj)

data = pd.DataFrame(obj)
data.columns = ['Name','x','y','x_down','y_down','x_Center','y_Center']
print(data)
# data.to_csv('Object_detection_list.csv')
# print(label)

