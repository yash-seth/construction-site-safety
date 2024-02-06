import os
from langchain.llms import GooglePalm
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders.csv_loader import CSVLoader
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


class LLMEvaluator:
    def relevancy_score(self, query, chain, embeddings):
        LLM_Response = chain(query)
        answer = LLM_Response['result']

        query_embeddings = embeddings.embed_query(query)
        answer_embeddings = embeddings.embed_query(answer)

        query_embeddings = np.array(query_embeddings)
        answer_embeddings = np.array(answer_embeddings)
        
        query_embeddings = query_embeddings.reshape(1, -1)
        answer_embeddings = answer_embeddings.reshape(1, -1)

        # Calculate cosine similarity
        similarity = cosine_similarity(query_embeddings, answer_embeddings)[0][0]
        print("Relevancy Score:", similarity)

    def relevancy_score_batch(self, queries, chain, embeddings):
        relevancyScore = 0
        for query in queries:
            LLM_Response = chain(query)
            answer = LLM_Response['result']

            query_embeddings = embeddings.embed_query(query)
            answer_embeddings = embeddings.embed_query(answer)

            query_embeddings = np.array(query_embeddings)
            answer_embeddings = np.array(answer_embeddings)
            
            query_embeddings = query_embeddings.reshape(1, -1)
            answer_embeddings = answer_embeddings.reshape(1, -1)

            # Calculate cosine similarity
            similarity = cosine_similarity(query_embeddings, answer_embeddings)[0][0]
            relevancyScore += similarity

        relevancyScore = relevancyScore / len(queries)
        print("Final Relevancy Score:", relevancyScore)

        relevancy_results_log = pd.read_csv(r'D:\My_Stuff\VIT-20BCE1789\Sem 8\Capstone\Work\Evaluate_LLM\relevancy_results.csv') 
        df = pd.DataFrame(columns=['RunID', 'Score'])
        df.loc[len(df.index)] = [len(relevancy_results_log)+1, relevancyScore]
        df.to_csv(r'D:\My_Stuff\VIT-20BCE1789\Sem 8\Capstone\Work\Evaluate_LLM\relevancy_results.csv', index=False, mode='a', header=False)

    def faithfulness(self, query):
        print("Inside faithfulness func")

    

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

evaluator = LLMEvaluator()


# evaluator.relevancy_score("Who all in the architecture department did not wear helmet?", chain, embeddings)

queries = ["Who all in the architecture department did not wear helmet?", "Who all didn't wear helmet in the worker logs?", "How many of the painting department didn't wear helmet AND face mask?", "List out the names of those who didn't wear helmet", "How many people are marked absent?"]


# running ten runs of relevancy metrics for the above 
for i in range(0, 10):
    evaluator.relevancy_score_batch(queries, chain, embeddings)