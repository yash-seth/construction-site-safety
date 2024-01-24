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
prompt_template = """"Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "status" and "timestamp" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

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
    df = pd.DataFrame(columns=['Timestamp', 'Name', 'Status'])

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
            print('I was here')
            try:
                # print('File: ', file.split(',')[1])
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

    # will track if helmet was found
    flag = False

    # loading file to model
    results = model("./Uploads/" + filename)

    # going over identified objects and their confidence values in the image
    for r in results:
        objMap = zip(r.boxes.cls, r.boxes.conf)
        for obj in objMap:
            if obj[0].item() == 0.0 and obj[1].item() >= 0.80:
                flag = True
                df.loc[len(df.index)] = [dt_string, worker_name, 'has helmet']
                print('has helmet')
                break
            else:
                continue

    # if no helmet was found with 80% or more confidence        
    if not flag:
        df.loc[len(df.index)] = [dt_string, worker_name, 'does not have helmet']
        print('does not have helmet')

    # append entry to worker log
    df.to_csv(r'./worker_log.csv', index=False, mode='a', header=False)

    return {
        'status': 'True' if flag else 'False'
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