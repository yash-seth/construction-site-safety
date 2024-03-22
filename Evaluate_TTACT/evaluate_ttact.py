from ultralytics import YOLO
import os
from os import listdir
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from sklearn.metrics import auc, plot_precision_recall_curve

def histogramEqualization(img_path, filename):
    img = cv2.imread(img_path, 0) 
    equ = cv2.equalizeHist(img)
    cv2.imwrite("D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/" + filename, equ) 

def gammaCorrection(img_path, filename):
    img = cv2.imread(img_path, 0) 
    gamma_corrected = np.array(255*(img / 255) ** 1.2, dtype = 'uint8') 
    cv2.imwrite("D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/" + filename, gamma_corrected) 

def gaussianBlurring(img_path, filename):
    img = cv2.imread(img_path, 0) 
    Gaussian = cv2.GaussianBlur(img, (7, 7), 0) 
    cv2.imwrite("D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/" + filename, Gaussian)

def histogramEqualizationNoise(img_path, filename):
    img = cv2.imread(img_path, 0) 
    equ = cv2.equalizeHist(img)
    noise = np.random.normal(0, 2, equ.shape)
    y = equ * 0.3 + 0.6
    equ = y + noise
    cv2.imwrite("D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/" + filename, equ) 

def gammaCorrectionNoise(img_path, filename):
    img = cv2.imread(img_path, 0) 
    gamma_corrected = np.array(255*(img / 255) ** 1.2, dtype = 'uint8') 
    noise = np.random.normal(0, 2, gamma_corrected.shape)
    y = gamma_corrected * 0.3 + 0.6
    gamma_corrected = y + noise
    cv2.imwrite("D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/" + filename, gamma_corrected) 

def gaussianBlurringNoise(img_path, filename):
    img = cv2.imread(img_path, 0) 
    Gaussian = cv2.GaussianBlur(img, (7, 7), 0) 
    noise = np.random.normal(0, 2, Gaussian.shape)
    y = Gaussian * 0.3 + 0.6
    Gaussian = y + noise
    cv2.imwrite("D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/" + filename, Gaussian)  

def normalizeRed(intensity):
    iI      = intensity
    minI    = 86
    maxI    = 230
    minO    = 0
    maxO    = 255
    iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)
    return iO

def normalizeGreen(intensity):
    iI      = intensity  
    minI    = 90
    maxI    = 225 
    minO    = 0
    maxO    = 255 
    iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)
    return iO

# Method to process the blue band of the image

def normalizeBlue(intensity):
    iI      = intensity   
    minI    = 100
    maxI    = 210
    minO    = 0
    maxO    = 255
    iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)
    return iO


def contrastStretching(img_path, filename):
    # Create an image object
    imageObject     = Image.open(img_path)    
    # Split the red, green and blue bands from the Image
    multiBands      = imageObject.split()
   
    # Apply point operations that does contrast stretching on each color band
    normalizedRedBand      = multiBands[0].point(normalizeRed)
    normalizedGreenBand    = multiBands[1].point(normalizeGreen)
    normalizedBlueBand     = multiBands[2].point(normalizeBlue)
   
    # Create a new image from the contrast stretched red, green and blue brands
    normalizedImage = Image.merge("RGB", (normalizedRedBand, normalizedGreenBand, normalizedBlueBand))
    normalizedImage.save("D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/" + filename) 

def contrastStretchingNoise(img_path, filename):
    # Create an image object
    imageObject     = Image.open(img_path)    
    # Split the red, green and blue bands from the Image
    multiBands      = imageObject.split()
   
    # Apply point operations that does contrast stretching on each color band
    normalizedRedBand      = multiBands[0].point(normalizeRed)
    normalizedGreenBand    = multiBands[1].point(normalizeGreen)
    normalizedBlueBand     = multiBands[2].point(normalizeBlue)
   
    # Create a new image from the contrast stretched red, green and blue brands
    normalizedImage = Image.merge("RGB", (normalizedRedBand, normalizedGreenBand, normalizedBlueBand))
    normalizedImage.save("D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/" + filename) 
    img = cv2.imread("D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/" + filename) 
    noise = np.random.normal(0, 2, img.shape)
    y = img * 0.3 + 0.6
    img = y + noise
    cv2.imwrite("D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/" + filename, img) 


# function to take in image and return the max confidence for the class required
def getMaxConfidence(results, label):
    maxConfidence = 0.0
    for r in results:
        objMap = zip(r.boxes.cls, r.boxes.conf)
        for obj in objMap:
            if obj[0].item() == label:
                maxConfidence = max(maxConfidence, obj[1].item())
    return maxConfidence * 100


model = YOLO('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/backend/yolov8_helmet_model.pt')

# # TTACT
# helmet_images = os.listdir('./images/Helmet/')
# non_helmet_images = os.listdir('./images/Non-Helmet/')
# df_TTACT = pd.DataFrame(columns=['Confidence Threshold', 'Precision', 'Recall'])
# print("TTACT:")
# for confidence_threshold in np.arange(0, 105, 5):
#     TP = 0
#     TN = 0
#     FP = 0
#     FN = 0
#     for path in helmet_images:
#         totalConfidence = 0
#         overallConfidence = 0
#         helmet_flag = True
#         for i in range(0,5,1):
#             # original image
#             if i == 0:
#                 results = model('./images/Helmet/' + path)
#                 totalConfidence += getMaxConfidence(results, 0.0)
#             # HE
#             if i == 1:
#                 histogramEqualization('./images/Helmet/' + path, path)
#                 results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 totalConfidence += getMaxConfidence(results, 0.0)
#                 os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#             # GC
#             if i == 2:
#                 gammaCorrection('./images/Helmet/' + path, path)
#                 results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 totalConfidence += getMaxConfidence(results, 0.0)
#                 os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#             # GB
#             if i == 3:
#                 gaussianBlurring('./images/Helmet/' + path, path)
#                 results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 totalConfidence += getMaxConfidence(results, 0.0)
#                 os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#             # CS
#             if i == 4:
#                 contrastStretching('./images/Helmet/' + path, path)
#                 results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                 totalConfidence += getMaxConfidence(results, 0.0)
#                 os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#         overallConfidence = totalConfidence / 5
#         if overallConfidence >= confidence_threshold:
#             pred = 1
#         else:
#             pred = 0

#         if pred == 1 and helmet_flag:
#             TP += 1
#         else:
#             FN += 1

#     totalConfidence = 0
#     overallConfidence = 0
#     helmet_flag = False
#     for path in non_helmet_images:
#         totalConfidence = 0
#         overallConfidence = 0
#         for i in range(0,5,1):
#             # original image
#             if i == 0:
#                 results = model('./images/Non-Helmet/' + path)
#                 totalConfidence += getMaxConfidence(results, 0.0)
#             # HE
#             if i == 1:
#                 histogramEqualization('./images/Non-Helmet/' + path, path)
#                 results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 totalConfidence += getMaxConfidence(results, 0.0)
#                 os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#             # GC
#             if i == 2:
#                 gammaCorrection('./images/Non-Helmet/' + path, path)
#                 results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 totalConfidence += getMaxConfidence(results, 0.0)
#                 os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#             # GB
#             if i == 3:
#                 gaussianBlurring('./images/Non-Helmet/' + path, path)
#                 results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 totalConfidence += getMaxConfidence(results, 0.0)
#                 os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#             # CS
#             if i == 4:
#                 contrastStretching('./images/Non-Helmet/' + path, path)
#                 results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                 totalConfidence += getMaxConfidence(results, 0.0)
#                 os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#         overallConfidence = totalConfidence / 5
#         if overallConfidence >= confidence_threshold:
#             pred = 1
#         else:
#             pred = 0

#         if pred == 0 and not helmet_flag:
#             TN += 1
#         else:
#             FP += 1

#     print(f"At {confidence_threshold}:")
#     print('TP:', TP)
#     print('FP:', FP)
#     print('TN:', TN)
#     print('FN:', FN)
#     precision = 0 if TP + FP == 0 else TP / (TP + FP)
#     recall = 0 if TP + FN == 0 else TP / (TP + FN)
#     print("Precision:", precision)
#     print("Recall:", recall)
#     df_TTACT.loc[len(df_TTACT.index)] = [confidence_threshold, precision, recall]

# df_TTACT.to_csv(r'./TTACT_PR.csv', index=False)



# # WM
# helmet_images = os.listdir('./images/Helmet/')
# non_helmet_images = os.listdir('./images/Non-Helmet/')
# df_wm = pd.DataFrame(columns=['Confidence Threshold', 'Precision', 'Recall'])
# print("Weighted Mean:")
# for confidence_threshold in np.arange(0, 105, 5):
#     TP = 0
#     TN = 0
#     FP = 0
#     FN = 0
#     for path in helmet_images:
#         totalConfidence = 0
#         overallConfidence = 0
#         helmet_flag = True
#         for i in range(0,3,1):
#             # original image
#             if i == 0:
#                 results = model('./images/Helmet/' + path)
#                 totalConfidence += 3 * getMaxConfidence(results, 0.0)
#             # HE
#             if i == 1:
#                 histogramEqualization('./images/Helmet/' + path, path)
#                 results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                 os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#             # GC
#             if i == 2:
#                 gammaCorrection('./images/Helmet/' + path, path)
#                 results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                 os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)


#         overallConfidence = totalConfidence / 5
#         if overallConfidence >= confidence_threshold:
#             pred = 1
#         else:
#             pred = 0

#         if pred == 1 and helmet_flag:
#             TP += 1
#         else:
#             FN += 1

#     totalConfidence = 0
#     overallConfidence = 0
#     helmet_flag = False
#     for path in non_helmet_images:
#         totalConfidence = 0
#         overallConfidence = 0
#         for i in range(0,3,1):
#             # original image
#             if i == 0:
#                 results = model('./images/Non-Helmet/' + path)
#                 totalConfidence += 3 * getMaxConfidence(results, 0.0)
#             # HE
#             if i == 1:
#                 histogramEqualization('./images/Non-Helmet/' + path, path)
#                 results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                 os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#             # GC
#             if i == 2:
#                 gammaCorrection('./images/Non-Helmet/' + path, path)
#                 results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                 os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
            

#         overallConfidence = totalConfidence / 5
#         if overallConfidence >= confidence_threshold:
#             pred = 1
#         else:
#             pred = 0

#         if pred == 0 and not helmet_flag:
#             TN += 1
#         else:
#             FP += 1

#     print(f"At {confidence_threshold}:")
#     print('TP:', TP)
#     print('FP:', FP)
#     print('TN:', TN)
#     print('FN:', FN)
#     precision = 0 if TP + FP == 0 else TP / (TP + FP)
#     recall = 0 if TP + FN == 0 else TP / (TP + FN)
#     print("Precision:", precision)
#     print("Recall:", recall)
#     df_wm.loc[len(df_wm.index)] = [confidence_threshold, precision, recall]

# df_wm.to_csv(r'./wm_PR.csv', index=False)


# # TTACT + Noise
# helmet_images = os.listdir('./images/Helmet/')
# non_helmet_images = os.listdir('./images/Non-Helmet/')
# df_TTACT_noise = pd.DataFrame(columns=['Confidence Threshold', 'Precision', 'Recall'])
# print("TTACT + Noise:")
# for confidence_threshold in np.arange(0, 105, 5):
#     TP = 0
#     TN = 0
#     FP = 0
#     FN = 0
#     for path in helmet_images:
#         totalConfidence = 0
#         overallConfidence = 0
#         helmet_flag = True
#         for i in range(0,5,1):
#             # original image
#             if i == 0:
#                 results = model('./images/Helmet/' + path)
#                 totalConfidence += getMaxConfidence(results, 0.0)
#             # HE
#             if i == 1:
#                 histogramEqualizationNoise('./images/Helmet/' + path, path)
#                 results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 totalConfidence += getMaxConfidence(results, 0.0)
#                 os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#             # GC
#             if i == 2:
#                 gammaCorrectionNoise('./images/Helmet/' + path, path)
#                 results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 totalConfidence += getMaxConfidence(results, 0.0)
#                 os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#             # GB
#             if i == 3:
#                 gaussianBlurringNoise('./images/Helmet/' + path, path)
#                 results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 totalConfidence += getMaxConfidence(results, 0.0)
#                 os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#             # CS
#             if i == 4:
#                 contrastStretchingNoise('./images/Helmet/' + path, path)
#                 results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                 totalConfidence += getMaxConfidence(results, 0.0)
#                 os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#         overallConfidence = totalConfidence / 5
#         if overallConfidence >= confidence_threshold:
#             pred = 1
#         else:
#             pred = 0

#         if pred == 1 and helmet_flag:
#             TP += 1
#         else:
#             FN += 1

#     totalConfidence = 0
#     overallConfidence = 0
#     helmet_flag = False
#     for path in non_helmet_images:
#         totalConfidence = 0
#         overallConfidence = 0
#         for i in range(0,5,1):
#             # original image
#             if i == 0:
#                 results = model('./images/Non-Helmet/' + path)
#                 totalConfidence += getMaxConfidence(results, 0.0)
#             # HE
#             if i == 1:
#                 histogramEqualizationNoise('./images/Non-Helmet/' + path, path)
#                 results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 totalConfidence += getMaxConfidence(results, 0.0)
#                 os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#             # GC
#             if i == 2:
#                 gammaCorrectionNoise('./images/Non-Helmet/' + path, path)
#                 results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 totalConfidence += getMaxConfidence(results, 0.0)
#                 os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#             # GB
#             if i == 3:
#                 gaussianBlurringNoise('./images/Non-Helmet/' + path, path)
#                 results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 totalConfidence += getMaxConfidence(results, 0.0)
#                 os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#             # CS
#             if i == 4:
#                 contrastStretchingNoise('./images/Non-Helmet/' + path, path)
#                 results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                 totalConfidence += getMaxConfidence(results, 0.0)
#                 os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#         overallConfidence = totalConfidence / 5
#         if overallConfidence >= confidence_threshold:
#             pred = 1
#         else:
#             pred = 0

#         if pred == 0 and not helmet_flag:
#             TN += 1
#         else:
#             FP += 1

#     print(f"At {confidence_threshold}:")
#     print('TP:', TP)
#     print('FP:', FP)
#     print('TN:', TN)
#     print('FN:', FN)
#     precision = 0 if TP + FP == 0 else TP / (TP + FP)
#     recall = 0 if TP + FN == 0 else TP / (TP + FN)
#     print("Precision:", precision)
#     print("Recall:", recall)
#     df_TTACT_noise.loc[len(df_TTACT_noise.index)] = [confidence_threshold, precision, recall]

# df_TTACT_noise.to_csv(r'./TTACT_noise_PR.csv', index=False)

# # new TTACT v1
# helmet_images = os.listdir('./images/Helmet/')
# non_helmet_images = os.listdir('./images/Non-Helmet/')
# df_TTACT_new = pd.DataFrame(columns=['Confidence Threshold', 'Precision', 'Recall'])
# print("New TTACT:")
# for confidence_threshold in np.arange(0, 105, 5):
#     TP = 0
#     TN = 0
#     FP = 0
#     FN = 0
#     for path in helmet_images:
#         tmpImg = cv2.imread('./images/Helmet/' + path)
#         meanPixelIntensityValue = tmpImg.mean(axis=0).mean()
#         # print(meanPixelIntensityValue)
#         if meanPixelIntensityValue < 80:
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = True
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Helmet/' + path)
#                     totalConfidence += 0.75 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 0.75 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 # GB
#                 if i == 3:
#                     gaussianBlurring('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                     totalConfidence += 0.75 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 0.75 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 1 and helmet_flag:
#                 TP += 1
#             else:
#                 FN += 1  

#         elif meanPixelIntensityValue >= 78 and meanPixelIntensityValue <= 178:
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = True
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Helmet/' + path)
#                     totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 1.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 0.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 # GB
#                 if i == 3:
#                     gaussianBlurring('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                     totalConfidence += 0.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 1.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 1 and helmet_flag:
#                 TP += 1
#             else:
#                 FN += 1 
#         else:
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = True
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Helmet/' + path)
#                     totalConfidence += 0.75 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 0.75 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 # GB
#                 if i == 3:
#                     gaussianBlurring('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                     totalConfidence += 0.75 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 0.75 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 1 and helmet_flag:
#                 TP += 1
#             else:
#                 FN += 1 

#     for path in non_helmet_images:
#         tmpImg = cv2.imread('./images/Non-Helmet/' + path)
#         meanPixelIntensityValue = tmpImg.mean(axis=0).mean()
#         # print(meanPixelIntensityValue)
#         if meanPixelIntensityValue < 80:
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = False
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Non-Helmet/' + path)
#                     totalConfidence += 0.75 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 0.75 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 # GB
#                 if i == 3:
#                     gaussianBlurring('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                     totalConfidence += 0.75 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 0.75 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 0 and not helmet_flag:
#                 TN += 1
#             else:
#                 FP += 1  

#         elif meanPixelIntensityValue >= 78 and meanPixelIntensityValue <= 178:
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = False
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Non-Helmet/' + path)
#                     totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 1.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 0.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 # GB
#                 if i == 3:
#                     gaussianBlurring('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                     totalConfidence += 0.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 1.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 0 and not helmet_flag:
#                 TN += 1
#             else:
#                 FP += 1
#         else:
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = False
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Non-Helmet/' + path)
#                     totalConfidence += 0.75 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 0.75 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 # GB
#                 if i == 3:
#                     gaussianBlurring('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                     totalConfidence += 0.75 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 0.75 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 0 and not helmet_flag:
#                 TN += 1
#             else:
#                 FP += 1

#     print(f"At {confidence_threshold}:")
#     print('TP:', TP)
#     print('FP:', FP)
#     print('TN:', TN)
#     print('FN:', FN)
#     precision = 0 if TP + FP == 0 else TP / (TP + FP)
#     recall = 0 if TP + FN == 0 else TP / (TP + FN)
#     print("Precision:", precision)
#     print("Recall:", recall)
#     df_TTACT_new.loc[len(df_TTACT_new.index)] = [confidence_threshold, precision, recall]

# df_TTACT_new.to_csv(r'./TTACT_new_PR.csv', index=False)


# # new TTACT v2
# helmet_images = os.listdir('./images/Helmet/')
# non_helmet_images = os.listdir('./images/Non-Helmet/')
# df_TTACT_new_v2 = pd.DataFrame(columns=['Confidence Threshold', 'Precision', 'Recall'])
# print("New TTACT v2:")
# for confidence_threshold in np.arange(0, 105, 5):
#     TP = 0
#     TN = 0
#     FP = 0
#     FN = 0
#     for path in helmet_images:
#         tmpImg = cv2.imread('./images/Helmet/' + path)
#         meanPixelIntensityValue = tmpImg.mean(axis=0).mean()
#         # print(meanPixelIntensityValue)
#         if meanPixelIntensityValue < 80:
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = True
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Helmet/' + path)
#                     totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 0.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 # GB
#                 if i == 3:
#                     gaussianBlurring('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                     totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 0.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 1 and helmet_flag:
#                 TP += 1
#             else:
#                 FN += 1  

#         elif meanPixelIntensityValue >= 78 and meanPixelIntensityValue <= 178:
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = True
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Helmet/' + path)
#                     totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 1.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 0.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 # GB
#                 if i == 3:
#                     gaussianBlurring('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                     totalConfidence += 0.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 1.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 1 and helmet_flag:
#                 TP += 1
#             else:
#                 FN += 1 
#         else:
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = True
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Helmet/' + path)
#                     totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 0.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 # GB
#                 if i == 3:
#                     gaussianBlurring('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                     totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 0.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 1 and helmet_flag:
#                 TP += 1
#             else:
#                 FN += 1 

#     for path in non_helmet_images:
#         tmpImg = cv2.imread('./images/Non-Helmet/' + path)
#         meanPixelIntensityValue = tmpImg.mean(axis=0).mean()
#         # print(meanPixelIntensityValue)
#         if meanPixelIntensityValue < 80:
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = False
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Non-Helmet/' + path)
#                     totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 0.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 # GB
#                 if i == 3:
#                     gaussianBlurring('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                     totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 0.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 0 and not helmet_flag:
#                 TN += 1
#             else:
#                 FP += 1  

#         elif meanPixelIntensityValue >= 78 and meanPixelIntensityValue <= 178:
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = False
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Non-Helmet/' + path)
#                     totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 1.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 0.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 # GB
#                 if i == 3:
#                     gaussianBlurring('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                     totalConfidence += 0.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 1.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 0 and not helmet_flag:
#                 TN += 1
#             else:
#                 FP += 1
#         else:
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = False
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Non-Helmet/' + path)
#                     totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 0.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 # GB
#                 if i == 3:
#                     gaussianBlurring('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                     totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 0.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 0 and not helmet_flag:
#                 TN += 1
#             else:
#                 FP += 1

#     print(f"At {confidence_threshold}:")
#     print('TP:', TP)
#     print('FP:', FP)
#     print('TN:', TN)
#     print('FN:', FN)
#     precision = 0 if TP + FP == 0 else TP / (TP + FP)
#     recall = 0 if TP + FN == 0 else TP / (TP + FN)
#     print("Precision:", precision)
#     print("Recall:", recall)
#     df_TTACT_new_v2.loc[len(df_TTACT_new_v2.index)] = [confidence_threshold, precision, recall]

# df_TTACT_new_v2.to_csv(r'./TTACT_new_v2_PR.csv', index=False)

# new TTACT v3
# helmet_images = os.listdir('./images/Helmet/')
# non_helmet_images = os.listdir('./images/Non-Helmet/')
# df_TTACT_new_v3 = pd.DataFrame(columns=['Confidence Threshold', 'Precision', 'Recall'])
# print("New TTACT v3:")
# for confidence_threshold in np.arange(0, 105, 5):
#     TP = 0
#     TN = 0
#     FP = 0
#     FN = 0
#     for path in helmet_images:
#         tmpImg = cv2.imread('./images/Helmet/' + path)
#         meanPixelIntensityValue = tmpImg.mean(axis=0).mean()
#         # print(meanPixelIntensityValue)
#         if meanPixelIntensityValue < 80:
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = True
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Helmet/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 0 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 # GB
#                 if i == 3:
#                     gaussianBlurring('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                     totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 0 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 1 and helmet_flag:
#                 TP += 1
#             else:
#                 FN += 1  

#         elif meanPixelIntensityValue >= 78 and meanPixelIntensityValue <= 178:
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = True
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Helmet/' + path)
#                     totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 1.75 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 0.25 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 # GB
#                 if i == 3:
#                     gaussianBlurring('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                     totalConfidence += 0.25 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 1.75 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 1 and helmet_flag:
#                 TP += 1
#             else:
#                 FN += 1 
#         else:
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = True
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Helmet/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 0 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 # GB
#                 if i == 3:
#                     gaussianBlurring('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                     totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 0 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 1 and helmet_flag:
#                 TP += 1
#             else:
#                 FN += 1 

#     for path in non_helmet_images:
#         tmpImg = cv2.imread('./images/Non-Helmet/' + path)
#         meanPixelIntensityValue = tmpImg.mean(axis=0).mean()
#         # print(meanPixelIntensityValue)
#         if meanPixelIntensityValue < 80:
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = False
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Non-Helmet/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 0 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 # GB
#                 if i == 3:
#                     gaussianBlurring('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                     totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 0 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 0 and not helmet_flag:
#                 TN += 1
#             else:
#                 FP += 1  

#         elif meanPixelIntensityValue >= 78 and meanPixelIntensityValue <= 178:
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = False
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Non-Helmet/' + path)
#                     totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 1.75 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 0.25 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 # GB
#                 if i == 3:
#                     gaussianBlurring('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                     totalConfidence += 0.25 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 1.75 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 0 and not helmet_flag:
#                 TN += 1
#             else:
#                 FP += 1
#         else:
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = False
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Non-Helmet/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 0 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 # GB
#                 if i == 3:
#                     gaussianBlurring('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                     totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 0 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 0 and not helmet_flag:
#                 TN += 1
#             else:
#                 FP += 1

#     print(f"At {confidence_threshold}:")
#     print('TP:', TP)
#     print('FP:', FP)
#     print('TN:', TN)
#     print('FN:', FN)
#     precision = 0 if TP + FP == 0 else TP / (TP + FP)
#     recall = 0 if TP + FN == 0 else TP / (TP + FN)
#     print("Precision:", precision)
#     print("Recall:", recall)
#     df_TTACT_new_v3.loc[len(df_TTACT_new_v3.index)] = [confidence_threshold, precision, recall]

# df_TTACT_new_v3.to_csv(r'./TTACT_new_v3_PR.csv', index=False)

# new TTACT v4
# brightness_counter = 0
# contrast_counter = 0
# helmet_images = os.listdir('./images/Helmet/')
# non_helmet_images = os.listdir('./images/Non-Helmet/')
# df_TTACT_new_v4 = pd.DataFrame(columns=['Confidence Threshold', 'Precision', 'Recall'])
# print("New TTACT v4:")
# for confidence_threshold in np.arange(0, 105, 5):
#     TP = 0
#     TN = 0
#     FP = 0
#     FN = 0
#     for path in helmet_images:
#         tmpImg = cv2.imread('./images/Helmet/' + path)
#         meanPixelIntensityValue = tmpImg.mean(axis=0).mean()
#         # print(meanPixelIntensityValue)
#         if meanPixelIntensityValue < 80:
#             brightness_counter += 1
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = True
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Helmet/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 0 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 # GB
#                 if i == 3:
#                     gaussianBlurring('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                     totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 0 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 1 and helmet_flag:
#                 TP += 1
#             else:
#                 FN += 1  

#         elif meanPixelIntensityValue >= 78 and meanPixelIntensityValue <= 178:
#             contrast_counter += 1
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = True
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Helmet/' + path)
#                     totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 1.75 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 0.25 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 # GB
#                 if i == 3:
#                     gaussianBlurring('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                     totalConfidence += 0.25 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 1.75 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 1 and helmet_flag:
#                 TP += 1
#             else:
#                 FN += 1 
#         else:
#             brightness_counter += 1
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = True
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Helmet/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 0 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 # GB
#                 if i == 3:
#                     gaussianBlurring('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                     totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 0 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 1 and helmet_flag:
#                 TP += 1
#             else:
#                 FN += 1 

#     for path in non_helmet_images:
#         tmpImg = cv2.imread('./images/Non-Helmet/' + path)
#         meanPixelIntensityValue = tmpImg.mean(axis=0).mean()
#         # print(meanPixelIntensityValue)
#         if meanPixelIntensityValue < 80:
#             brightness_counter += 1
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = False
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Non-Helmet/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 0 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 # GB
#                 if i == 3:
#                     gaussianBlurring('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                     totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 0 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 0 and not helmet_flag:
#                 TN += 1
#             else:
#                 FP += 1  

#         elif meanPixelIntensityValue >= 78 and meanPixelIntensityValue <= 178:
#             contrast_counter += 1
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = False
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Non-Helmet/' + path)
#                     totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 1.75 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 0.25 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 # GB
#                 if i == 3:
#                     gaussianBlurring('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                     totalConfidence += 0.25 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 1.75 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 0 and not helmet_flag:
#                 TN += 1
#             else:
#                 FP += 1
#         else:
#             brightness_counter += 1
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = False
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Non-Helmet/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 0 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 # GB
#                 if i == 3:
#                     gaussianBlurring('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                     totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 0 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 0 and not helmet_flag:
#                 TN += 1
#             else:
#                 FP += 1

#     print(f"At {confidence_threshold}:")
#     print('TP:', TP)
#     print('FP:', FP)
#     print('TN:', TN)
#     print('FN:', FN)
#     precision = 0 if TP + FP == 0 else TP / (TP + FP)
#     recall = 0 if TP + FN == 0 else TP / (TP + FN)
#     print("Precision:", precision)
#     print("Recall:", recall)
#     df_TTACT_new_v4.loc[len(df_TTACT_new_v4.index)] = [confidence_threshold, precision, recall]

# df_TTACT_new_v4.to_csv(r'./TTACT_new_v4_PR.csv', index=False)

# v5
# brightness_counter = 0
# contrast_counter = 0
# helmet_images = os.listdir('./images/Helmet/')
# non_helmet_images = os.listdir('./images/Non-Helmet/')
# df_TTACT_new_v5 = pd.DataFrame(columns=['Confidence Threshold', 'Precision', 'Recall'])
# print("New TTACT v5:")
# for confidence_threshold in np.arange(0, 105, 5):
#     TP = 0
#     TN = 0
#     FP = 0
#     FN = 0
#     for path in helmet_images:
#         tmpImg = cv2.imread('./images/Helmet/' + path)
#         meanPixelIntensityValue = tmpImg.mean(axis=0).mean()
#         # print(meanPixelIntensityValue)
#         if meanPixelIntensityValue < 80:
#             brightness_counter += 1
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = True
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Helmet/' + path)
#                     totalConfidence += 2.5 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 0 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 # GB
#                 if i == 3:
#                     gaussianBlurring('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                     totalConfidence += 0.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 0 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 1 and helmet_flag:
#                 TP += 1
#             else:
#                 FN += 1  

#         elif meanPixelIntensityValue >= 78 and meanPixelIntensityValue <= 178:
#             contrast_counter += 1
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = True
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Helmet/' + path)
#                     totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 0 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 # GB
#                 if i == 3:
#                     gaussianBlurring('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                     totalConfidence += 0 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 1 and helmet_flag:
#                 TP += 1
#             else:
#                 FN += 1 
#         else:
#             brightness_counter += 1
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = True
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Helmet/' + path)
#                     totalConfidence += 2.5 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 0 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 # GB
#                 if i == 3:
#                     gaussianBlurring('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                     totalConfidence += 0.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 0 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 1 and helmet_flag:
#                 TP += 1
#             else:
#                 FN += 1 

#     for path in non_helmet_images:
#         tmpImg = cv2.imread('./images/Non-Helmet/' + path)
#         meanPixelIntensityValue = tmpImg.mean(axis=0).mean()
#         # print(meanPixelIntensityValue)
#         if meanPixelIntensityValue < 80:
#             brightness_counter += 1
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = False
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Non-Helmet/' + path)
#                     totalConfidence += 2.5 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 0 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 # GB
#                 if i == 3:
#                     gaussianBlurring('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                     totalConfidence += 0.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 0 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 0 and not helmet_flag:
#                 TN += 1
#             else:
#                 FP += 1  

#         elif meanPixelIntensityValue >= 78 and meanPixelIntensityValue <= 178:
#             contrast_counter += 1
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = False
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Non-Helmet/' + path)
#                     totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 0 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 # GB
#                 if i == 3:
#                     gaussianBlurring('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                     totalConfidence += 0 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 0 and not helmet_flag:
#                 TN += 1
#             else:
#                 FP += 1
#         else:
#             brightness_counter += 1
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = False
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Non-Helmet/' + path)
#                     totalConfidence += 2.5 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 0 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                 # GB
#                 if i == 3:
#                     gaussianBlurring('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                     totalConfidence += 0.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GB/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 0 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 0 and not helmet_flag:
#                 TN += 1
#             else:
#                 FP += 1

#     print(f"At {confidence_threshold}:")
#     print('TP:', TP)
#     print('FP:', FP)
#     print('TN:', TN)
#     print('FN:', FN)
#     precision = 0 if TP + FP == 0 else TP / (TP + FP)
#     recall = 0 if TP + FN == 0 else TP / (TP + FN)
#     print("Precision:", precision)
#     print("Recall:", recall)
#     df_TTACT_new_v5.loc[len(df_TTACT_new_v5.index)] = [confidence_threshold, precision, recall]

# df_TTACT_new_v5.to_csv(r'./TTACT_new_v5_PR.csv', index=False)

# print("Brightness Coutner:", brightness_counter)
# print("Contrast Coutner:", contrast_counter)

# # v6
# brightness_counter = 0
# contrast_counter = 0
# helmet_images = os.listdir('./images/Helmet/')
# non_helmet_images = os.listdir('./images/Non-Helmet/')
# df_TTACT_new_v6 = pd.DataFrame(columns=['Confidence Threshold', 'Precision', 'Recall'])
# print("New TTACT v6:")
# for confidence_threshold in np.arange(0, 105, 5):
#     TP = 0
#     TN = 0
#     FP = 0
#     FN = 0
#     for path in helmet_images:
#         tmpImg = cv2.imread('./images/Helmet/' + path)
#         meanPixelIntensityValue = tmpImg.mean(axis=0).mean()
#         # print(meanPixelIntensityValue)
#         if meanPixelIntensityValue < 80:
#             brightness_counter += 1
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = True
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Helmet/' + path)
#                     totalConfidence += 3 * getMaxConfidence(results, 0.0)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 1 and helmet_flag:
#                 TP += 1
#             else:
#                 FN += 1  

#         elif meanPixelIntensityValue >= 78 and meanPixelIntensityValue <= 178:
#             contrast_counter += 1
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = True
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Helmet/' + path)
#                     totalConfidence += 3 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 1 and helmet_flag:
#                 TP += 1
#             else:
#                 FN += 1 
#         else:
#             brightness_counter += 1
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = True
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Helmet/' + path)
#                     totalConfidence += 3 * getMaxConfidence(results, 0.0)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)


#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 1 and helmet_flag:
#                 TP += 1
#             else:
#                 FN += 1 

#     for path in non_helmet_images:
#         tmpImg = cv2.imread('./images/Non-Helmet/' + path)
#         meanPixelIntensityValue = tmpImg.mean(axis=0).mean()
#         # print(meanPixelIntensityValue)
#         if meanPixelIntensityValue < 80:
#             brightness_counter += 1
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = False
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Non-Helmet/' + path)
#                     totalConfidence += 3 * getMaxConfidence(results, 0.0)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 0 and not helmet_flag:
#                 TN += 1
#             else:
#                 FP += 1  

#         elif meanPixelIntensityValue >= 78 and meanPixelIntensityValue <= 178:
#             contrast_counter += 1
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = False
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Non-Helmet/' + path)
#                     totalConfidence += 3 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                 # CS
#                 if i == 4:
#                     contrastStretching('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)
#                     totalConfidence += 1 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/CS/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 0 and not helmet_flag:
#                 TN += 1
#             else:
#                 FP += 1
#         else:
#             brightness_counter += 1
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = False
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Non-Helmet/' + path)
#                     totalConfidence += 3 * getMaxConfidence(results, 0.0)
#                 if i == 2:
#                     gammaCorrection('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 0 and not helmet_flag:
#                 TN += 1
#             else:
#                 FP += 1

#     print(f"At {confidence_threshold}:")
#     print('TP:', TP)
#     print('FP:', FP)
#     print('TN:', TN)
#     print('FN:', FN)
#     precision = 0 if TP + FP == 0 else TP / (TP + FP)
#     recall = 0 if TP + FN == 0 else TP / (TP + FN)
#     print("Precision:", precision)
#     print("Recall:", recall)
#     df_TTACT_new_v6.loc[len(df_TTACT_new_v6.index)] = [confidence_threshold, precision, recall]

# df_TTACT_new_v6.to_csv(r'./TTACT_new_v6_PR.csv', index=False)

# v7
# brightness_counter = 0
# contrast_counter = 0
# helmet_images = os.listdir('./images/Helmet/')
# non_helmet_images = os.listdir('./images/Non-Helmet/')
# df_TTACT_new_v7 = pd.DataFrame(columns=['Confidence Threshold', 'Precision', 'Recall'])
# print("New TTACT v7:")
# for confidence_threshold in np.arange(0, 105, 5):
#     TP = 0
#     TN = 0
#     FP = 0
#     FN = 0
#     for path in helmet_images:
#         tmpImg = cv2.imread('./images/Helmet/' + path)
#         meanPixelIntensityValue = tmpImg.mean(axis=0).mean()
#         # print(meanPixelIntensityValue)
#         if meanPixelIntensityValue < 80:
#             brightness_counter += 1
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = True
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Helmet/' + path)
#                     totalConfidence += 3 * getMaxConfidence(results, 0.0)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 1 and helmet_flag:
#                 TP += 1
#             else:
#                 FN += 1  

#         elif meanPixelIntensityValue >= 78 and meanPixelIntensityValue <= 178:
#             contrast_counter += 1
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = True
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Helmet/' + path)
#                     totalConfidence += 3 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 1 and helmet_flag:
#                 TP += 1
#             else:
#                 FN += 1 
#         else:
#             brightness_counter += 1
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = True
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Helmet/' + path)
#                     totalConfidence += 3 * getMaxConfidence(results, 0.0)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)


#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 1 and helmet_flag:
#                 TP += 1
#             else:
#                 FN += 1 

#     for path in non_helmet_images:
#         tmpImg = cv2.imread('./images/Non-Helmet/' + path)
#         meanPixelIntensityValue = tmpImg.mean(axis=0).mean()
#         # print(meanPixelIntensityValue)
#         if meanPixelIntensityValue < 80:
#             brightness_counter += 1
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = False
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Non-Helmet/' + path)
#                     totalConfidence += 3 * getMaxConfidence(results, 0.0)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 0 and not helmet_flag:
#                 TN += 1
#             else:
#                 FP += 1  

#         elif meanPixelIntensityValue >= 78 and meanPixelIntensityValue <= 178:
#             contrast_counter += 1
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = False
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Non-Helmet/' + path)
#                     totalConfidence += 3 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 0 and not helmet_flag:
#                 TN += 1
#             else:
#                 FP += 1
#         else:
#             brightness_counter += 1
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = False
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Non-Helmet/' + path)
#                     totalConfidence += 3 * getMaxConfidence(results, 0.0)
#                 if i == 2:
#                     gammaCorrection('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 0 and not helmet_flag:
#                 TN += 1
#             else:
#                 FP += 1

#     print(f"At {confidence_threshold}:")
#     print('TP:', TP)
#     print('FP:', FP)
#     print('TN:', TN)
#     print('FN:', FN)
#     precision = 0 if TP + FP == 0 else TP / (TP + FP)
#     recall = 0 if TP + FN == 0 else TP / (TP + FN)
#     print("Precision:", precision)
#     print("Recall:", recall)
#     df_TTACT_new_v7.loc[len(df_TTACT_new_v7.index)] = [confidence_threshold, precision, recall]

# df_TTACT_new_v7.to_csv(r'./TTACT_new_v7_PR.csv', index=False)


# v8
# brightness_counter = 0
# contrast_counter = 0
# helmet_images = os.listdir('./images/Helmet/')
# non_helmet_images = os.listdir('./images/Non-Helmet/')
# df_TTACT_new_v8 = pd.DataFrame(columns=['Confidence Threshold', 'Precision', 'Recall'])
# print("New TTACT v8:")
# for confidence_threshold in np.arange(0, 105, 5):
#     TP = 0
#     TN = 0
#     FP = 0
#     FN = 0
#     for path in helmet_images:
#         tmpImg = cv2.imread('./images/Helmet/' + path)
#         meanPixelIntensityValue = tmpImg.mean(axis=0).mean()
#         # print(meanPixelIntensityValue)
#         if meanPixelIntensityValue < 80:
#             brightness_counter += 1
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = True
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Helmet/' + path)
#                     totalConfidence += 2.5 * getMaxConfidence(results, 0.0)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 1 and helmet_flag:
#                 TP += 1
#             else:
#                 FN += 1  

#         elif meanPixelIntensityValue >= 78 and meanPixelIntensityValue <= 178:
#             contrast_counter += 1
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = True
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Helmet/' + path)
#                     totalConfidence += 2.5 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 2.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 1 and helmet_flag:
#                 TP += 1
#             else:
#                 FN += 1 
#         else:
#             brightness_counter += 1
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = True
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Helmet/' + path)
#                     totalConfidence += 2.5 * getMaxConfidence(results, 0.0)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)


#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 1 and helmet_flag:
#                 TP += 1
#             else:
#                 FN += 1 

#     for path in non_helmet_images:
#         tmpImg = cv2.imread('./images/Non-Helmet/' + path)
#         meanPixelIntensityValue = tmpImg.mean(axis=0).mean()
#         # print(meanPixelIntensityValue)
#         if meanPixelIntensityValue < 80:
#             brightness_counter += 1
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = False
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Non-Helmet/' + path)
#                     totalConfidence += 2.5 * getMaxConfidence(results, 0.0)
#                 # GC
#                 if i == 2:
#                     gammaCorrection('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 0 and not helmet_flag:
#                 TN += 1
#             else:
#                 FP += 1  

#         elif meanPixelIntensityValue >= 78 and meanPixelIntensityValue <= 178:
#             contrast_counter += 1
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = False
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Non-Helmet/' + path)
#                     totalConfidence += 2.5 * getMaxConfidence(results, 0.0)
#                 # HE
#                 if i == 1:
#                     histogramEqualization('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
#                     totalConfidence += 2.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 0 and not helmet_flag:
#                 TN += 1
#             else:
#                 FP += 1
#         else:
#             brightness_counter += 1
#             totalConfidence = 0
#             overallConfidence = 0
#             helmet_flag = False
#             for i in range(0,5,1):
#                 # original image
#                 if i == 0:
#                     results = model('./images/Non-Helmet/' + path)
#                     totalConfidence += 2.5 * getMaxConfidence(results, 0.0)
#                 if i == 2:
#                     gammaCorrection('./images/Non-Helmet/' + path, path)
#                     results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
#                     totalConfidence += 2.5 * getMaxConfidence(results, 0.0)
#                     os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)

#             overallConfidence = totalConfidence / 5
#             if overallConfidence >= confidence_threshold:
#                 pred = 1
#             else:
#                 pred = 0

#             if pred == 0 and not helmet_flag:
#                 TN += 1
#             else:
#                 FP += 1

#     print(f"At {confidence_threshold}:")
#     print('TP:', TP)
#     print('FP:', FP)
#     print('TN:', TN)
#     print('FN:', FN)
#     precision = 0 if TP + FP == 0 else TP / (TP + FP)
#     recall = 0 if TP + FN == 0 else TP / (TP + FN)
#     print("Precision:", precision)
#     print("Recall:", recall)
#     df_TTACT_new_v8.loc[len(df_TTACT_new_v8.index)] = [confidence_threshold, precision, recall]

# df_TTACT_new_v8.to_csv(r'./TTACT_new_v8_PR.csv', index=False)


# v9
brightness_counter = 0
contrast_counter = 0
helmet_images = os.listdir('./images/Helmet/')
non_helmet_images = os.listdir('./images/Non-Helmet/')
df_TTACT_new_v9 = pd.DataFrame(columns=['Confidence Threshold', 'Precision', 'Recall'])
print("New TTACT v9:")
for confidence_threshold in np.arange(0, 105, 5):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for path in helmet_images:
        tmpImg = cv2.imread('./images/Helmet/' + path)
        meanPixelIntensityValue = tmpImg.mean(axis=0).mean()
        # print(meanPixelIntensityValue)
        if meanPixelIntensityValue < 80:
            brightness_counter += 1
            totalConfidence = 0
            overallConfidence = 0
            helmet_flag = True
            for i in range(0,5,1):
                # original image
                if i == 0:
                    results = model('./images/Helmet/' + path)
                    totalConfidence += 4 * getMaxConfidence(results, 0.0)
                # GC
                if i == 2:
                    gammaCorrection('./images/Helmet/' + path, path)
                    results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
                    totalConfidence += 1 * getMaxConfidence(results, 0.0)
                    os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)

            overallConfidence = totalConfidence / 5
            if overallConfidence >= confidence_threshold:
                pred = 1
            else:
                pred = 0

            if pred == 1 and helmet_flag:
                TP += 1
            else:
                FN += 1  

        elif meanPixelIntensityValue >= 78 and meanPixelIntensityValue <= 178:
            contrast_counter += 1
            totalConfidence = 0
            overallConfidence = 0
            helmet_flag = True
            for i in range(0,5,1):
                # original image
                if i == 0:
                    results = model('./images/Helmet/' + path)
                    totalConfidence += 4 * getMaxConfidence(results, 0.0)
                # HE
                if i == 1:
                    histogramEqualization('./images/Helmet/' + path, path)
                    results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
                    totalConfidence += 1 * getMaxConfidence(results, 0.0)
                    os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)

            overallConfidence = totalConfidence / 5
            if overallConfidence >= confidence_threshold:
                pred = 1
            else:
                pred = 0

            if pred == 1 and helmet_flag:
                TP += 1
            else:
                FN += 1 
        else:
            brightness_counter += 1
            totalConfidence = 0
            overallConfidence = 0
            helmet_flag = True
            for i in range(0,5,1):
                # original image
                if i == 0:
                    results = model('./images/Helmet/' + path)
                    totalConfidence += 4 * getMaxConfidence(results, 0.0)
                # GC
                if i == 2:
                    gammaCorrection('./images/Helmet/' + path, path)
                    results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
                    totalConfidence += 1 * getMaxConfidence(results, 0.0)
                    os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)


            overallConfidence = totalConfidence / 5
            if overallConfidence >= confidence_threshold:
                pred = 1
            else:
                pred = 0

            if pred == 1 and helmet_flag:
                TP += 1
            else:
                FN += 1 

    for path in non_helmet_images:
        tmpImg = cv2.imread('./images/Non-Helmet/' + path)
        meanPixelIntensityValue = tmpImg.mean(axis=0).mean()
        # print(meanPixelIntensityValue)
        if meanPixelIntensityValue < 80:
            brightness_counter += 1
            totalConfidence = 0
            overallConfidence = 0
            helmet_flag = False
            for i in range(0,5,1):
                # original image
                if i == 0:
                    results = model('./images/Non-Helmet/' + path)
                    totalConfidence += 4 * getMaxConfidence(results, 0.0)
                # GC
                if i == 2:
                    gammaCorrection('./images/Non-Helmet/' + path, path)
                    results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
                    totalConfidence += 1 * getMaxConfidence(results, 0.0)
                    os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)

            overallConfidence = totalConfidence / 5
            if overallConfidence >= confidence_threshold:
                pred = 1
            else:
                pred = 0

            if pred == 0 and not helmet_flag:
                TN += 1
            else:
                FP += 1  

        elif meanPixelIntensityValue >= 78 and meanPixelIntensityValue <= 178:
            contrast_counter += 1
            totalConfidence = 0
            overallConfidence = 0
            helmet_flag = False
            for i in range(0,5,1):
                # original image
                if i == 0:
                    results = model('./images/Non-Helmet/' + path)
                    totalConfidence += 4 * getMaxConfidence(results, 0.0)
                # HE
                if i == 1:
                    histogramEqualization('./images/Non-Helmet/' + path, path)
                    results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)
                    totalConfidence += 1 * getMaxConfidence(results, 0.0)
                    os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/HE/' + path)

            overallConfidence = totalConfidence / 5
            if overallConfidence >= confidence_threshold:
                pred = 1
            else:
                pred = 0

            if pred == 0 and not helmet_flag:
                TN += 1
            else:
                FP += 1
        else:
            brightness_counter += 1
            totalConfidence = 0
            overallConfidence = 0
            helmet_flag = False
            for i in range(0,5,1):
                # original image
                if i == 0:
                    results = model('./images/Non-Helmet/' + path)
                    totalConfidence += 4 * getMaxConfidence(results, 0.0)
                if i == 2:
                    gammaCorrection('./images/Non-Helmet/' + path, path)
                    results = model('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)
                    totalConfidence += 1 * getMaxConfidence(results, 0.0)
                    os.remove('D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/Evaluate_TTACT/images/Transformations/GC/' + path)

            overallConfidence = totalConfidence / 5
            if overallConfidence >= confidence_threshold:
                pred = 1
            else:
                pred = 0

            if pred == 0 and not helmet_flag:
                TN += 1
            else:
                FP += 1

    print(f"At {confidence_threshold}:")
    print('TP:', TP)
    print('FP:', FP)
    print('TN:', TN)
    print('FN:', FN)
    precision = 0 if TP + FP == 0 else TP / (TP + FP)
    recall = 0 if TP + FN == 0 else TP / (TP + FN)
    print("Precision:", precision)
    print("Recall:", recall)
    df_TTACT_new_v9.loc[len(df_TTACT_new_v9.index)] = [confidence_threshold, precision, recall]

df_TTACT_new_v9.to_csv(r'./TTACT_new_v9_PR.csv', index=False)


print("Brightness Coutner:", brightness_counter)
print("Contrast Coutner:", contrast_counter)