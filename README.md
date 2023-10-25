## BCS_PDF_READER
A language processing pipeline for retrieval-based question answering using the Langchain library, uploaded on HuggingFace.

This pipeline is live on: https://huggingface.co/spaces/bcs-iitk/PDF-ChatBot-BCS

## Table of Contents 

- [Overview](#overview)
- [Features](#features)
- [How to use the gradio interface](#howtousethegradiointerface)
- [Prerequisites](#prerequisites)

## Overview

This repository contains a Python script for building a retrieval-based question-answering system using the Langchain library. It offers a comprehensive language processing pipeline designed to help you answer questions based on textual data stored in PDF documents. The pipeline includes the following key components:

1. **PDF Document Loader**: This model utilizes the `OnlinePDFLoader` from Langchain to load and extract text content from PDF documents. It prepares the PDF content for further processing.

2. **Text Splitter**: The `RecursiveCharacterTextSplitter` is responsible for splitting large text content into manageable chunks, ensuring efficient and accurate text processing. It uses various separators to intelligently segment the text.

3. **Embeddings**: The chatbot employs the `HuggingFaceHubEmbeddings` package to compute embeddings for text data. These embeddings capture the semantic information of the text, which is vital for retrieval-based question answering.

4. **Vector Stores**: To store and efficiently retrieve embeddings, the chatbot utilizes `FAISS`, which is a high-performance similarity search library from Facebook AI. It offers fast, approximate similarity search capabilities, enabling quick retrieval of relevant documents.

5. **Question Answering (QA)**: The pipeline incorporates a RetrievalQA component that allows users to ask questions based on the embeddings of the text data. It retrieves and ranks documents that contain relevant information to answer the user's query.

The primary use case for this pipeline is to process PDF documents, generate embeddings, and enable users to ask questions about the document content. Whether you're conducting research, analyzing reports, or searching for information in a large document collection, this system can assist in extracting meaningful answers efficiently.

## Features
- Interact with PDF documents using a user-friendly interface.
- Ask questions and receive answers from PDF content.
- Load PDF documents or select from pre-loaded books.

## How to use the gradio interface
- In the Gradio interface, you can load a PDF document or choose from pre-loaded books. 
- Type your questions in the chat window and hit Enter.
- The chatbot will provide answers based on the content of the PDF document.
  
## Prerequisites

If you wish to use this chatbot on your own machine, ensure you have the following dependencies installed:

- Python 3.8+
- pip
- Required Python packages (install them using `pip`):
  - Langchain
  - PyPDF
  - Sentence-Transformers
  - Faiss-CPU
  - NumPy
  - Pandas
