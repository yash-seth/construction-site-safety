from flask import Flask, request
import os
from os import listdir
from langchain.llms import GooglePalm
from flask_cors import CORS, cross_origin
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from PIL import Image 
from ultralytics import YOLO
import pandas as pd
from datetime import datetime

# set up env for LLM Chain
llm = GooglePalm(google_api_key="AIzaSyCm-45dqF12sh65lga0ERhSTWYXneFSt8k", temperature = 0.7)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./gen-lang-client-0503785992-0790e07a62c7.json"

# create embeddings object - to create and interpret embeddings
embeddings = HuggingFaceInstructEmbeddings()

# load vectorDB for embeddings of the worker logs
vectorDB = FAISS.load_local(r"D:\My_Stuff\VIT-20BCE1789\Sem 8\Capstone\Work\faiss_index", embeddings)
retriever = vectorDB.as_retriever()

# prompt for the LLM queries and response - to prevent hallucinations
prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "status" section in the source document context without making much changes.
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

@app.route('/query', methods=['POST', 'OPTIONS'])
@cross_origin()
def queryLogs():
    res = ''
    res = chain(request.json['query'])
    return {
        "response": res['result'] if res['result'] else 'Response not available!'
    }


@app.route('/upload', methods=['GET', 'POST', 'OPTIONS'])
@cross_origin(origin='*')
def upload():
    file = request.files['file']
    model = YOLO('./yolov8_helmet_model.pt')
    df = pd.DataFrame(columns=['Timestamp', 'Name', 'Status'])

    filename = request.form['filename']
    fname = filename.split('.')[0].split('_')[0]
    lname = filename.split('.')[0].split('_')[1]
    worker_name = fname + ' ' + lname

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    
    file.save("./Uploads/" + filename)
    flag = False
    results = model("./Uploads/" + filename)
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
    if not flag:
        df.loc[len(df.index)] = [dt_string, worker_name, 'does not have helmet']
        print('does not have helmet')

    df.to_csv(r'./worker_log.csv', index=False, mode='a', header=False)

    return {
        'status': 'True' if flag else 'False'
    }

# dummy post endpoint
@app.route('/dummy_endpoint', methods=['GET', 'POST', 'OPTIONS'])
@cross_origin(origin='*')
def dummy_endpoint():
    file = request.files['file']
    # file.save("./Uploads/abc.png")
    # Image.open(request.files['file'])
    return {
        'Name':"geek", 
        "Age":"22", 
        "programming":"python"
    }