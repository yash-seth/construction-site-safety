import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from ultralytics.utils.plotting import Annotator
import numpy as np

# Define path to video file
source = './combined_short.mp4'

# return percentage of the helmet [a1, a2, b1, b2] which lies within the bounding box of the identified person [x1, x2, y1, y2]
def getOverlap(x1, x2, y1, y2, a1, a2, b1, b2):
    # checking for non-overlapping case
    if a1 > x2 or x1 > a2 or b1 > y2 or y1 > b2:
        return 0

    top_coordinate = [max(a1, x1), max(b1, y1)]
    bottom_coordinate = [min(a2, x2), min(b2, y2)]

    AreaOfOverlap = (bottom_coordinate[0] - top_coordinate[0]) * (bottom_coordinate[1] - top_coordinate[1])

    # total area of helmet
    AreaOfHelmet = (b2 - b1) * (a2 - a1)

    # area of overlap over total area of helmet gives % of helmet that lies in the bounding box of the person
    PercentageOverlap = (AreaOfOverlap / AreaOfHelmet) * 100
    return PercentageOverlap

vid = cv2.VideoCapture(source)
property_id = int(cv2.CAP_PROP_FRAME_COUNT)  
length = int(cv2.VideoCapture.get(vid, property_id)) 
print("Number of frames in the video: ", length) 

frameCounter = 0
while True:
    frameCounter += 1
    successful, frame = vid.read()
    if successful:
        print("\nFrame ", frameCounter)
        # identifying the helmets in the current frame
        Helmet_Model = YOLO('../backend/yolov8_helmet_model.pt')
        Helmet_Results = Helmet_Model(frame)
        helmet_boxes = Helmet_Results[0].boxes.xyxy.tolist()
        helmet_classes = Helmet_Results[0].boxes.cls.tolist()
        helmet_names = Helmet_Results[0].names
        helmet_confidences = Helmet_Results[0].boxes.conf.tolist()

        # identifying the people, PPE kits and face masks in the current frame
        PPE_FaceMask_Model = YOLO('../backend/multiclass-yolov8.pt')
        PPE_FaceMask_Results = PPE_FaceMask_Model(frame)
        PPE_FaceMask_boxes = PPE_FaceMask_Results[0].boxes.xyxy.tolist()
        PPE_FaceMask_classes = PPE_FaceMask_Results[0].boxes.cls.tolist()
        PPE_FaceMask_names = PPE_FaceMask_Results[0].names
        PPE_FaceMask_confidences = PPE_FaceMask_Results[0].boxes.conf.tolist()
        
        print()

        for box, cls, conf in zip(PPE_FaceMask_boxes, PPE_FaceMask_classes, PPE_FaceMask_confidences):
            x1, y1, x2, y2 = box
            confidence = conf
            detected_class = cls
            name = PPE_FaceMask_names[int(cls)]

            # person is class 5.0
            if detected_class == 5.0:
                PercentageOverlap = 0
                for box1, cls1, conf1 in zip(helmet_boxes, helmet_classes, helmet_confidences):
                    # helmet is class 0.0
                    if cls1 != 0.0: continue
                    a1, b1, a2, b2 = box1
                    PercentageOverlap = max(PercentageOverlap, getOverlap(x1, x2, y1, y2, a1, a2, b1, b2))
                print("Overlap percentage: ", PercentageOverlap)
                if PercentageOverlap < 80:
                    print("Worker without helmet spotted!")
                    print(f"Bounding box of worker without helmet: (x1, y1): ({x1}, {y1}), (x2, y2): ({x2}, {y2})")
    else:
        break
        
vid.release()
cv2.destroyAllWindows()

# to visualize detected objects in video feed directly
# model = YOLO('../backend/yolov8_helmet_model.pt')

# results = model(source, save=True, line_width = 2, conf=0.5)

# model = YOLO('../backend/multiclass-yolov8.pt')

# results = model(source, save=True, line_width = 2, conf=0.5)