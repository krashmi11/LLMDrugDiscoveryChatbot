from langchain.llms import GooglePalm
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import CSVLoader,DirectoryLoader
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()
import os

llm=GooglePalm(api_key=os.environ["GOOGLE_API_KEY"],temperature=0.7)

#embedding using Hugging face
embeddings = HuggingFaceInstructEmbeddings(
    query_instruction="Represent the query for retrieval: "
)
vector_db_path="faiss_index"

def create_vector_db():
    #loading csv
    # loader = CSVLoader(file_path='Bioactivity1.csv',source_column='prompt')
    loader = CSVLoader(file_path='Bioactivity1.csv',source_column='prompt')
    documents=loader.load()
    db = FAISS.from_documents(documents=documents, embedding=embeddings)
    db.save_local(vector_db_path)

def qa_chain():
    new_db = FAISS.load_local(vector_db_path, embeddings)
    retriever = new_db.as_retriever(search_kwargs={"k":3},score_threshold = 0.7)
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}


    chain = RetrievalQA.from_chain_type(llm=llm,
                            chain_type="stuff",
                            retriever=retriever,
                            input_key="query",
                            return_source_documents=True,
                            chain_type_kwargs=chain_type_kwargs)
    return chain
import warnings

if __name__=="__main__":
    # warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
    # print(chain("determine characteristics of canonical smile CSc1nc(-c2ccc(Cl)cc2)nn1C(=O)N(C)C ?"))
    # create_vector_db()
    chain=qa_chain()
