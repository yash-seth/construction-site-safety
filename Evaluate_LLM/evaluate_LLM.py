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


class RAGEvaluator:
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

        return similarity

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

        return relevancyScore

    def faithfulness_score(self, query, chain, embeddings):
        LLM_Response = chain(query)
        answer = LLM_Response['result']
        answer_embeddings = embeddings.embed_query(answer)
        answer_embeddings = np.array(answer_embeddings)
        answer_embeddings = answer_embeddings.reshape(1, -1)
        
        context = LLM_Response['source_documents']
        faithfulness_score = 0
        for key, doc in enumerate(context):
            doc_content = doc.page_content
            doc_content_embedding = embeddings.embed_query(doc_content)
            doc_content_embedding = np.array(doc_content_embedding)
            doc_content_embedding = doc_content_embedding.reshape(1, -1)
            similarity = cosine_similarity(doc_content_embedding, answer_embeddings)[0][0]
            faithfulness_score += similarity
        
        faithfulnessScore = faithfulness_score / len(context)
        return faithfulnessScore

    def faithfulness_score_batch(self, queries, chain, embeddings):
        final_faithfulness_score = 0
        for query in queries:
            LLM_Response = chain(query)
            answer = LLM_Response['result']
            answer_embeddings = embeddings.embed_query(answer)
            answer_embeddings = np.array(answer_embeddings)
            answer_embeddings = answer_embeddings.reshape(1, -1)
            
            context = LLM_Response['source_documents']
            faithfulness_score = 0
            for key, doc in enumerate(context):
                doc_content = doc.page_content
                doc_content_embedding = embeddings.embed_query(doc_content)
                doc_content_embedding = np.array(doc_content_embedding)
                doc_content_embedding = doc_content_embedding.reshape(1, -1)
                similarity = cosine_similarity(doc_content_embedding, answer_embeddings)[0][0]
                faithfulness_score += similarity
            
            avg_faithfulness_score = faithfulness_score / len(context)
            final_faithfulness_score += avg_faithfulness_score
        
        final_faithfulness_score = final_faithfulness_score / len(queries)

        return final_faithfulness_score

    def context_recall(self, query, chain, embeddings):
        LLM_Response = chain(query)
        answer = LLM_Response['result']

        # creating query embeddings
        query_embeddings = embeddings.embed_query(query)
        query_embeddings = np.array(query_embeddings)
        query_embeddings = query_embeddings.reshape(1, -1)

        # recalled docs
        recalledDocs = LLM_Response['source_documents']
        recalledDocsContent = []
        for docs in recalledDocs:
            docString = ''
            timeStampCounter = 0
            for col in docs.page_content.split('\n'):
                if timeStampCounter == 1:
                    docString += col.split(':')[1].strip() + ':' + col.split(':')[2].strip() + ':' + col.split(':')[3].strip() + ' '
                else:
                    docString += col.split(':')[1].strip() + ' '
                timeStampCounter += 1
            recalledDocsContent.append(docString)
        
        # get all docs in the worker logs
        worker_log = pd.read_csv(r'D:\My_Stuff\VIT-20BCE1789\Sem 8\Capstone\Work\frontend\backend\\worker_log.csv')
        allDocs = []
        columns = ['LogID','Timestamp','WorkerID','Name','Department','Helmet-Status','PPE-Status','FaceMask-Status','Attendance']
        for ind in worker_log.index:
            log = ""
            for col in columns:
                log += str(worker_log[col][ind]).strip() + " "
            allDocs.append(log)

        # find the actually relevant docs from all docs by comparing query with allDocs and taking out those with cosine > 0.5
        relevantDocs = []
        for doc in allDocs:
            doc_embedding = embeddings.embed_query(doc)
            doc_embedding = np.array(doc_embedding)
            doc_embedding = doc_embedding.reshape(1, -1)
            similarity = cosine_similarity(doc_embedding, query_embeddings)[0][0]
            # print(doc, similarity)
            if similarity > 0.80:
                relevantDocs.append(doc)

        # check how many of relevant docs were actually recalled out of the relevant docs
        counter = 0
        for doc in recalledDocsContent:
            if doc in relevantDocs:
                counter +=1

        recall = counter / len(relevantDocs)
        return [counter, len(relevantDocs), recall]
    
    def context_recall_batch(self, queries, chain, embeddings):
        recallScore = 0
        for query in queries:
            LLM_Response = chain(query)
            answer = LLM_Response['result']

            # creating query embeddings
            query_embeddings = embeddings.embed_query(query)
            query_embeddings = np.array(query_embeddings)
            query_embeddings = query_embeddings.reshape(1, -1)

            # recalled docs
            recalledDocs = LLM_Response['source_documents']
            recalledDocsContent = []
            for docs in recalledDocs:
                docString = ''
                timeStampCounter = 0
                for col in docs.page_content.split('\n'):
                    if timeStampCounter == 1:
                        docString += col.split(':')[1].strip() + ':' + col.split(':')[2].strip() + ':' + col.split(':')[3].strip() + ' '
                    else:
                        docString += col.split(':')[1].strip() + ' '
                    timeStampCounter += 1
                recalledDocsContent.append(docString)
            
            # get all docs in the worker logs
            worker_log = pd.read_csv(r'D:\My_Stuff\VIT-20BCE1789\Sem 8\Capstone\Work\frontend\backend\\worker_log.csv')
            allDocs = []
            columns = ['LogID','Timestamp','WorkerID','Name','Department','Helmet-Status','PPE-Status','FaceMask-Status','Attendance']
            for ind in worker_log.index:
                log = ""
                for col in columns:
                    log += str(worker_log[col][ind]).strip() + " "
                allDocs.append(log)

            # find the actually relevant docs from all docs by comparing query with allDocs and taking out those with cosine > 0.5
            relevantDocs = []
            for doc in allDocs:
                doc_embedding = embeddings.embed_query(doc)
                doc_embedding = np.array(doc_embedding)
                doc_embedding = doc_embedding.reshape(1, -1)
                similarity = cosine_similarity(doc_embedding, query_embeddings)[0][0]
                # print(doc, similarity)
                if similarity > 0.80:
                    relevantDocs.append(doc)

            # check how many of relevant docs were actually recalled out of the relevant docs
            counter = 0
            for doc in recalledDocsContent:
                if doc in relevantDocs:
                    counter +=1
            
            recall = 0
            if len(relevantDocs) != 0:
                recall = counter / len(relevantDocs)
            recallScore += recall
        return recallScore / len(queries)

    def generateMetrics(self, queries, chain, embeddings, metric, num_of_runs):
        if metric == 'faithfulness':
            for i in range(0, num_of_runs):
                run_faithfulness_score = self.faithfulness_score_batch(queries, chain, embeddings)
                df = pd.DataFrame(columns=['RunID', 'Score'])
                df.loc[len(df.index)] = [i+1, run_faithfulness_score]
                df.to_csv(r'D:\My_Stuff\VIT-20BCE1789\Sem 8\Capstone\Work\frontend\Evaluate_LLM\faithfulness_results.csv', index=False, mode='a', header=False)

        elif metric == 'relevancy':
            for i in range(0, num_of_runs):
                run_relevancy_score = self.relevancy_score_batch(queries, chain, embeddings)
                df = pd.DataFrame(columns=['RunID', 'Score'])
                df.loc[len(df.index)] = [i+1, run_relevancy_score]
                df.to_csv(r'D:\My_Stuff\VIT-20BCE1789\Sem 8\Capstone\Work\frontend\Evaluate_LLM\relevancy_results.csv', index=False, mode='a', header=False)


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

# instatntiating the RAG Evaluator Object
evaluator = RAGEvaluator()

# add the queries you want to generate metrics for
queries = ["Who all in the architecture department did not wear helmet?", "Who all didn't wear helmet in the worker logs?", "How many of the painting department didn't wear helmet AND face mask?", "List out the names of those who didn't wear helmet", "How many people are marked absent?"]

# running ten runs of faithfulness metrics for the above 
# evaluator.generateMetrics(queries, chain, embeddings, 'faithfulness', 10)
evaluator.generateMetrics(queries, chain, embeddings, 'relevancy', 10)
