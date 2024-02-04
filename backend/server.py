from flask import Flask, request
import os
from os import listdir
from langchain.llms import GooglePalm
from flask_cors import CORS, cross_origin
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import base64, binascii
import uuid
import cv2
import numpy as np
from PIL import Image
from langchain.document_loaders.csv_loader import CSVLoader

# set up env for LLM Chain
llm = GooglePalm(google_api_key="AIzaSyCm-45dqF12sh65lga0ERhSTWYXneFSt8k", temperature = 0.7)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./gen-lang-client-0503785992-0790e07a62c7.json"

# create embeddings object - to create and interpret embeddings
embeddings = HuggingFaceInstructEmbeddings()

# load vectorDB for embeddings of the worker logs
vectorDB = FAISS.load_local(r"D:\My_Stuff\VIT-20BCE1789\Sem 8\Capstone\Work\faiss_index", embeddings)
retriever = vectorDB.as_retriever()

# prompt for the LLM queries and response - to prevent hallucinations
prompt_template = """"The given context is that of a worker attendance log. It contains the entries of workers along with the status of     their respective protective gear along with the timestamp of when they logged in and other information about their work.
    Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from 'LogID', 'Timestamp', "WorkerID", 'Name', "Department", 'Helmet-Status', 'PPE-Status', 'FaceMask-Status' and 'Attendance' section in the source document context in the form of whole english sentences will all required parts of speech. If the requirement is mathematical or to list out, go over all the relevant documents in the context and then perform the necessary operations. Provide accurate results for mathematical and/or scenarios which require people names or other attributes to be listed out.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer. 
    
    If more than one relevant documents exist for the same person, create your response around the most latest 'Timestamp' from the given documents in the context.

    CONTEXT: {context}

    QUESTION: {question}"""

PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )


chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=retriever,
                                    input_key="query",
                                    return_source_documents=True,
                                    chain_type_kwargs={"prompt": PROMPT})


# colour transformations for applying Test Time Augmentation (TTA) technique to improve object detection

def histogramEqualization(img_path, filename):
    img = cv2.imread(img_path, 0) 
    equ = cv2.equalizeHist(img)
    cv2.imwrite("./Transformations/HistogramEqualization/" + filename, equ) 

def gammaCorrection(img_path, filename):
    img = cv2.imread(img_path, 0) 
    gamma_corrected = np.array(255*(img / 255) ** 1.2, dtype = 'uint8') 
    cv2.imwrite("./Transformations/GammaCorrected/" + filename, gamma_corrected) 

def gaussianBlurring(img_path, filename):
    img = cv2.imread(img_path, 0) 
    Gaussian = cv2.GaussianBlur(img, (7, 7), 0) 
    cv2.imwrite("./Transformations/GaussianBlurred/" + filename, Gaussian) 

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
    normalizedImage.save("./Transformations/ContrastStretched/" + filename) 


# function to take in image and return the max confidence for the class required
def getMaxConfidence(results, label):
    maxConfidence = 0.0
    for r in results:
        objMap = zip(r.boxes.cls, r.boxes.conf)
        for obj in objMap:
            if obj[0].item() == label:
                maxConfidence = max(maxConfidence, obj[1].item())
    return maxConfidence * 100

# set up flask app with CORS to allow sending and receiving data
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/')
@cross_origin()
def hello():
   return "Hello World!"

@app.route('/syncLogs', methods=['GET', 'OPTIONS'])
@cross_origin(origin='*')
def syncLogs():
    print("Syncing....")
    loader = CSVLoader(file_path="./worker_log.csv", source_column="Name")
    logs = loader.load()    
    vectorDB = FAISS.from_documents(documents=logs, embedding=embeddings)
    vectorDB.save_local(r"D:\My_Stuff\VIT-20BCE1789\Sem 8\Capstone\Work\faiss_index")
    return {
                'status': 'Sync Successful'
            }

@app.route('/clearLogs', methods=['GET', 'OPTIONS'])
@cross_origin(origin='*')
def clearLogs():
    print("Clearing Logs....")
    try:
        os.remove("worker_log.csv")
        df = pd.DataFrame(columns=['LogID', 'Timestamp', "WorkerID", 'Name', "Department", 'Helmet-Status', 'PPE-Status', 'FaceMask-Status', 'Attendance'])
        df.to_csv(r'./worker_log.csv', index=False,)
        return {
                    'status': 'Logs were successfully cleared.'
                }
    except:
        return {
                    'status': 'There was an error in clearing the logs.'
                }
   

@app.route('/query', methods=['POST', 'OPTIONS'])
@cross_origin()
def queryLogs():
    vectorDB = FAISS.load_local(r"D:\My_Stuff\VIT-20BCE1789\Sem 8\Capstone\Work\faiss_index", embeddings)
    retriever = vectorDB.as_retriever()
    chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=retriever,
                                    input_key="query",
                                    return_source_documents=True,
                                    chain_type_kwargs={"prompt": PROMPT})
    res = ''
    res = chain(request.json['query'])
    return {
        "response": res['result'] if res['result'] else 'Response not available!'
    }


@app.route('/upload', methods=['GET', 'POST', 'OPTIONS'])
@cross_origin(origin='*')
def upload():

    image_source = request.form['source']

    if image_source == 'upload':
        try:
            file = request.files['file']
        except:
            return {
                'status': 'Error'
            }

    elif image_source == 'capture':
        try:
            file = request.form['file']
        except:
            return {
                'status': 'Error'
            }

    model = YOLO('./yolov8_helmet_model.pt')
    df = pd.DataFrame(columns=['LogID', 'Timestamp', "WorkerID", 'Name', "Department", 'Helmet-Status', 'PPE-Status', 'FaceMask-Status', 'Attendance'])

    # get file name (from worker name)
    filename = request.form['filename']
    fname = filename.split('.')[0].split('_')[0]
    lname = filename.split('.')[0].split('_')[1]
    worker_name = fname + ' ' + lname

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    
    # save file to pass to model
    try:
        if image_source == 'upload':
            file.save("./Uploads/" + filename)
        elif image_source == 'capture':
            try:
                base64_string = file.split(',')[1]
                image = base64.b64decode(base64_string, validate=True)
                file_to_save = "./Uploads/" + filename
                with open(file_to_save, "wb") as f:
                    f.write(image)
            except binascii.Error as e:
                print(e)
    except:
        return {
                'status': 'Error'
            }

    workerID = request.form['workerID']
    workerDep = request.form['workerDep']

    # worker result list object
    logID = uuid.uuid4() # can serve as unique identifier for each worker log
    worker_result = [logID, dt_string, workerID, worker_name, workerDep]


    # generating image transformations for uploaded image to apply TTA technique
    histogramEqualization("./Uploads/" + filename, filename)
    gammaCorrection("./Uploads/" + filename, filename)
    gaussianBlurring("./Uploads/" + filename, filename)
    contrastStretching("./Uploads/" + filename, filename)

    # paths to the transformed images and original image
    img_paths = ["./Uploads/", "./Transformations/ContrastStretched/", "./Transformations/GaussianBlurred/", "./Transformations/GammaCorrected/", "./Transformations/HistogramEqualization/"]

    # will track if helmet was found
    helmet_flag = False

    # none type colour transformation
    totalConfidence = 0
    for path in img_paths:
        results = model(path + filename)
        totalConfidence += getMaxConfidence(results, 0.0)

    allModelConfidence = totalConfidence / 500
    if allModelConfidence >= 0.75:
        helmet_flag = True
    else:
        helmet_flag = False

    print("All model Confidence for helmet: ", allModelConfidence)
    if helmet_flag:
        worker_result.append('has helmet')
        print('has helmet')
    # # if no helmet was found with 80% or more confidence        
    if not helmet_flag:
        worker_result.append('does not have helmet')
        print('does not have helmet')

    # multiclass model for face mask and ppe kit
        
    # 0: Hardhat
    # 1: Mask
    # 2: NO-Hardhat
    # 3: NO-Mask
    # 4: NO-Safety Vest
    # 5: Person
    # 6: Safety Cone
    # 7: Safety Vest
    # 8: Machinery
    # 9: Vehicle

    print("PPE Kit Results:")
    model = YOLO('./multiclass-yolov8.pt')
    ppe_flag = False

    totalConfidence = 0
    for path in img_paths:
        results = model(path + filename)
        totalConfidence += getMaxConfidence(results, 7.0)

    allModelConfidence = totalConfidence / 500
    if allModelConfidence >= 0.75:
        ppe_flag = True
    else:
        ppe_flag = False

    print("All model Confidence for PPE: ", allModelConfidence)
    # if no PPE Kit was found with 80% or more confidence    
    if ppe_flag:
        worker_result.append('has PPE kit')
        print('has PPE kit')    
    if not ppe_flag:
        worker_result.append('does not have PPE Kit')
        print('does not have PPE Kit')

    
    print("Face Mask Results:")
    model = YOLO('./multiclass-yolov8.pt')
    mask_flag = False

    totalConfidence = 0
    for path in img_paths:
        results = model(path + filename)
        totalConfidence += getMaxConfidence(results, 1.0)

    allModelConfidence = totalConfidence / 500
    if allModelConfidence >= 0.75:
        mask_flag = True
    else:
        mask_flag = False

    print("All model Confidence for mask: ", allModelConfidence)
    # if no face mask was found with 80% or more confidence    
    if mask_flag:
        worker_result.append('has face mask')
        print('has face mask')    
    if not mask_flag:
        worker_result.append('does not have face mask')
        print('does not have face mask')

    # marking overall attendance based on identified safety gear status
    attendance_flag = False
    if helmet_flag and mask_flag and ppe_flag:
        worker_result.append("Present")
        attendance_flag = True
    else:
        worker_result.append("Absent")

    df.loc[len(df.index)] = worker_result
    # append entry to worker log
    df.to_csv(r'./worker_log.csv', index=False, mode='a', header=False)
    df.to_csv(r'D:/My_Stuff/VIT-20BCE1789/Sem 8/Capstone/Work/frontend/attendance-frontend/src/Components/Admin/worker_log.csv', index=False, mode='a', header=False)

    # remove stored images
    for path in img_paths:
        os.remove(path + filename)

    return {
        'status': 'complete',
        'helmet_status': helmet_flag,
        'ppe_status': ppe_flag,
        'mask_status': mask_flag,
        'attendance_status': attendance_flag
    }

# dummy post endpoint
@app.route('/dummy_endpoint', methods=['GET', 'POST', 'OPTIONS'])
@cross_origin(origin='*')
def dummy_endpoint():
    return {
        'Name':"geek", 
        "Age":"22", 
        "programming":"python"
    }