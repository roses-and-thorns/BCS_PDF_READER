import gradio as gr
from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from datasets import load_dataset
import os

key = os.environ.get('RLS')
os.environ["HUGGINGFACEHUB_API_TOKEN"] = key

import sentence_transformers
import faiss

def loading_pdf():
    return "Loading..."

def pdf_changes(pdf_doc):
    
    loader = OnlinePDFLoader(pdf_doc.name)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=64,
        separators=['\n\n', '\n', '(?=>\. )', ' ', '']
    )
    docs  = text_splitter.split_documents(pages)
    embeddings = HuggingFaceHubEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    
    llm=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":1, "max_length":1000000})
    global qa 
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",  retriever=db.as_retriever(search_kwargs={"k": 3}))
    return "Ready"

def book_changes(book):
    db = FAISS.load_local( book , embeddings = HuggingFaceHubEmbeddings() )
    llm=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":1, "max_length":1000000})
    global qa 
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",  retriever=db.as_retriever(search_kwargs={"k": 3}))
    return "Ready"



def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def bot(history):
    response = infer(history[-1][0])
    history[-1][1] = response['result']
    return history

def infer(question):
    
    query = question
    result = qa({"query": query})

    return result

css="""
#col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
"""

title = """
<div style="text-align: center;max-width: 700px;">
    <h1>Chat with PDF</h1>   
"""


with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(title)
        with gr.Column():
            pdf_doc = gr.File(label="Load a PDF", file_types=['.pdf'], type="file")
            load_pdf = gr.Button("Load PDF")
            Books = gr.Dropdown(label="Books", choices=[("Harry Potter and the Philosopher's Stone","Book1")] )
            langchain_status = gr.Textbox(label="Status", placeholder="", interactive=False)
        chatbot = gr.Chatbot([], elem_id="chatbot").style(height=350)
        question = gr.Textbox(label="Question", placeholder="Type your question and hit Enter ")
        submit_btn = gr.Button("Send message") 
    Books.change(book_changes, inputs=[Books], outputs=[langchain_status], queue=False)
    load_pdf.click(pdf_changes, inputs=[pdf_doc], outputs=[langchain_status], queue=False)
    question.submit(add_text, [chatbot, question], [chatbot, question]).then(
        bot, chatbot, chatbot
    )
    submit_btn.click(add_text, [chatbot, question], [chatbot, question]).then(
        bot, chatbot, chatbot
    )

demo.launch()
