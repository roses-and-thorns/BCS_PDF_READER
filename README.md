# BCS_PDF_READER
A language processing pipeline for retrieval-based question answering using the Langchain library.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [License](#license)

## Overview

This repository contains a Python script for building a retrieval-based question-answering system using the Langchain library. It offers a comprehensive language processing pipeline designed to help you answer questions based on textual data stored in PDF documents. The pipeline includes the following key components:

- **Document Loading**: It leverages the PyPDFLoader to extract text content from PDF documents.

- **Text Splitting**: The RecursiveCharacterTextSplitter is used to split large text content into manageable chunks, ensuring efficient processing.

- **Embeddings**: The HuggingFaceEmbeddings module is employed to compute embeddings for the text data. These embeddings capture the semantic information of the text, which is crucial for retrieval-based question answering.

- **Vector Stores**: FAISS (Facebook AI Similarity Search) is utilized to store and efficiently retrieve embeddings. FAISS provides fast, approximate similarity search capabilities, which are essential for quickly finding relevant documents.

- **Question Answering (QA)**: The pipeline incorporates a RetrievalQA component that allows users to ask questions based on the embeddings of the text data. It retrieves and ranks documents that contain relevant information to answer the user's query.

The primary use case for this pipeline is to process PDF documents, generate embeddings, and enable users to ask questions about the document content. Whether you're conducting research, analyzing reports, or searching for information in a large document collection, this system can assist in extracting meaningful answers efficiently.


## Prerequisites

Before you can use this, ensure you have the following dependencies installed:

- Python 3.7+
- Pip
- Required Python packages (install them using `pip`):
  - Langchain
  - PyPDF
  - Sentence-Transformers
  - Faiss-CPU
  - NumPy
  - Pandas

 ## License
 
