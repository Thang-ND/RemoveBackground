import cv2
import numpy as np

def load_model():
    #load model YOLO
    net = cv2.dnn.readNet('P1_Detection/yolov4.weights', 'P1_Detection/cfg/yolov4.cfg')
    classes=None
    with open('P1_Detection/data/coco.names','r') as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0]-1]for i in net.getUnconnectedOutLayers()]
    # for i in net.getUnconnectedOutLayers():
    #     print(i)
    # print(layer_names)
    # random color for draw object
    colors = np.random.randint(0, 255, size=(43, 3)).astype(float) #color for draw objects

    return net, classes, colors, output_layers


def load_image(img_path='P1_Detection/data/guitar2.jpg'):
    image = cv2.imread(img_path)
    height,width = image.shape[:2]
#     if height >= 1000 or width >= 1000:
#         image = cv2.resize(image, None, fx=0.5, fy=0.5)

    height,width = image.shape[:2]
    return image, height, width


def detect_objects(image, net, output_layers):
    blob = cv2.dnn.blobFromImage(image, scalefactor=1/255, size=(416,416), mean=(0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    return outputs


def get_boxes(outputs, height, width):
    boxes = [] #list contain information for object(x_left_conner, y_left_conner, width, height)
    confidence_score = [] #confidence score
    class_ids = [] #index class in classes
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            classid = np.argmax(scores)
            conf = scores[classid]
            if conf > 0.5:
                x_center = int(detect[0]*width)
                y_center = int(detect[1]*height)
                w = int(detect[2]*width)
                h = int(detect[3]*height)
                x_left_conner = int(x_center - w/2)
                y_left_conner = int(y_center - h/2)
                boxes.append([x_left_conner, y_left_conner, w, h])
                class_ids.append(classid)
                confidence_score.append(float(conf)) #need to convert to float,if not,error built-in appear
    return boxes, confidence_score, class_ids


def draw_labels(boxs, confidences, colors, classes, class_ids, image):
    confidence_threshold = 0.5
    nms_threshold = 0.4
    indexs = cv2.dnn.NMSBoxes(boxs, confidences, confidence_threshold, nms_threshold)
    #print(indexs)
    boxes = []
    id = 0
    for index in indexs:
        i = index[0]
        x, y, w, h = boxs[i]
        boxes.append(boxs[i])
        #labelYolo = classes[class_ids[i]]+': {:.3f}'.format(confidences[i])
        labelYolo = classes[class_ids[i]]+ "- ID: " + str(id)
        #print(labelYolo)
        color = colors[class_ids[i]]
    
        '''
        Note: In this cv2.rectangle(), parameter(x,y),(x + w, y + h) must be integer.
        '''
        z = 0
        if ((x - z) < 0 or (y - z) < 0 or (x + w + z) < 0 or (y + h + z) < 0): continue
        image_crop = image[y - z : y + h + z, x - z : x + w + z]
        # image_crop = image[y : y + h, x : x + w]
        # image_crop = image.crop((x, y, x + w, y + h))
        # name, index = classification.predict(image_crop)
        # color = colors[index[i]]
        cv2.rectangle(image, (x - z, y - z), (x + w + z, y + h + z), color, 2)

        #label=str(name)+",id:"+str(index)
        cv2.putText(image, labelYolo, (x - 50, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        id += 1

    return image, boxes

def get_url(img_path):
    net, classes, colors, output_layers = load_model()
    image, height, width = load_image(img_path)
    outputs = detect_objects(image, net, output_layers)
    boxes, confidences, class_ids = get_boxes(outputs, height, width)
    output_image, new_boxes = draw_labels(boxes, confidences, colors, classes, class_ids, image)
    return output_image, new_boxes
