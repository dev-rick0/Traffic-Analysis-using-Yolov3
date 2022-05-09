#This project requires opencv and the yolov3.weight be present before it is run 



import cv2
import numpy as np
# Loading function for YOLO Algorithm weights and configuration file
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# List to store the object names
classes = []

# Opening the file containing the object names
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

# Loading target video file for detection
cap = cv2.VideoCapture('test1.mp4')

# Loading target image file for detection
# img = cv2.imread('truck.jpg')

while True:
    _, img = cap.read()
    height, width, _ = img.shape
    # Preparing the image to accepted form of the model
    # Normalizing the image by dividing the pixel value by 255
    # Resizing image to 416 by 416 to fit the size of yolov3
    # Value to be in RGB order, currently in BGR
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    # passing blob as input into network
    net.setInput(blob)

    # getting information we need for the projections, bounding boxes, predictions
    output_layer_names = net.getUnconnectedOutLayersNames()

    # passing into a forward function to do a forward pass-through of the pre trained model to obtain output
    layerOutputs = net.forward(output_layer_names)

    # lists for extracting bounding boxes, storing the confidence level and store predicted classes id
    boxes = []
    confidences = []
    class_ids = []


# getting info from the layers output
    for output in layerOutputs:
        # extracting info from each of the output
        for detection in output:
            # To extract the classes prediction starting from the sixth element to the end class
            scores = detection[5:]
            # identifying classes with highest scores
            class_id = np.argmax(scores)
            # identify maximum value from the class identified with highest score value
            confidence = scores[class_id]
            # setting minimum confidence/probability level
            if confidence > 0.5:
                # multiplying so as to extract them to full size after previous resize
                # centre x coordinates of object
                center_x = int(detection[0]*width)
                # centre y coordinates of object
                center_y = int(detection[1]*height)
                # width of object
                w = int(detection[2]*width)
                # length of object
                h = int(detection[3]*height)
                # extracting upper left and right positions for use in opencv
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                # applying extracted information to corresponding lists
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

# no of objects being detected
            print(len(boxes))
# NonMaximumSuppression function to keep highest score boxes only 0.4 is threshhold
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


    font = cv2.FONT_HERSHEY_PLAIN
# random color for each object detected
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

# fixes the attribute error of tuple
    if len(indexes) > 0:
    # looping over all objects detected
        for i in indexes.flatten(0):
            x, y, w, h = boxes[i]
            # labelling the detected objects
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, label + "" + confidence, (x, y+20), font, 2, (255, 255, 255), 2)


    cv2.imshow('truck', img)
    key = cv2.waitKey(1)
# terminating using escape key
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
