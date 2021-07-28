"""
    Script runs program by using command-line:
    python run.py -i input_path -o output_path -a type_of_action [[optional] -s index_of_object]

    Example: 
    Th1: python run.py -i input/people.jpg -o output_P2 -a remove-bg (thực hiện chỉ tách background)
    Th2: 
        Step 1: python run.py -i input/people.jpg -o output_P1 -a detect (thực hiện detect object trả về bounding box)
        Step 2: python run.py -i input/people.jpg -o output_P2 -a remove-bg -s 0,1,4,8 (thực hiện tách bg cho các đối tượng đã được chọn 0, 1, 4, 8)
"""
import argparse
import cv2
from PIL import Image
import os 
import numpy as np
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='input_path', type=str, help='path of input images')
parser.add_argument('-o', dest='output_path', type=str, help='path of output images')
parser.add_argument('-a', dest='action', type=str, help='action to call service')
parser.add_argument('-s', dest='selection', default=None, type=str, help='index of chosen object')
args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path
action = args.action
#print(os.sep)
filename_origin = input_path.split('/') 
filename = filename_origin[-1].split('.')
if action == 'detect':
    from P1_Detection import run
    output_image, boxes = run.get_url(input_path)
    output_file = output_path + '/' + filename[0] + '.' + filename[1] 
    cv2.imwrite(output_file, output_image)

    output_boxes = output_path + '/' + filename[0] + '.txt'
    with open(output_boxes, 'w') as f:
        for bb in boxes:
            f.write(str(bb)+'\n')
    
    # Do something with output image
elif action == 'remove-bg':
    from P2_RemoveBG import inference
    selection = args.selection
    if selection == None:
        output_image = inference.get_url(input_path)
        file_name = filename[0] + '.png'
        output_image.save(output_path+'/'+file_name)
    else:
        ii32 = np.iinfo(np.int32(10))
        selection_list = selection.split(",")
        selection_list = list(map(int, selection_list))
        x_min, y_min = ii32.max, ii32.max
        x_max, y_max = ii32.min, ii32.min
        output_bboxes = output_path + '/' + filename[0] + '.txt'
        boxes = []
        with open(output_bboxes, 'r') as f:
            while(True):
                bb = f.readline()
                if not bb: break
                bb = bb[1:-2].split(',')
                bb = list(map(int, bb))
                boxes.append(bb)
        
        for s in selection_list:
            x_min = min(boxes[s][0], x_min)
            y_min = min(boxes[s][1], y_min)      
            x_max = max(boxes[s][0] + boxes[s][2], x_max)  
            y_max = max(boxes[s][1] + boxes[s][3], y_max)  

        print(x_min, y_min, x_max, y_max)
        img = cv2.imread(input_path)
        h, w = img.shape[:2]
        z = 10
        x_min = max(0, x_min - z)
        y_min = max(0, y_min - z)
        x_max = min(w, x_max + z)
        y_max = min(h, y_max + z)
        img = img[y_min: y_max, x_min:x_max]
        cv2.imwrite("input_crop/" + filename[0]+ "." + filename[1], img)
        output_image = inference.get_url("input_crop/" + filename[0]+ "." + filename[1])
        file_name = filename[0] + '.png'
        output_image.save(output_path+'/'+file_name)

        # Do something with output image
else:
    print('Not a suitable action!')
    exit()
