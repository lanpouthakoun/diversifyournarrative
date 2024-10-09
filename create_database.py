import os
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from pinecone import Pinecone
import streamlit as st
from langchain_pinecone import PineconeVectorStore
from uuid import uuid4

from langchain_core.documents import Document


api_key_ = st.secrets["PINECONE_API_KEY"]
openai_key = st.secrets["OPENAI_API_KEY"]

os.environ["OPENAI_API_KEY"] =openai_key



pc = Pinecone(api_key=api_key_)
index = pc.Index("curriculum")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = PineconeVectorStore(index=index, embedding=embeddings)




# Load environment variables


pdf_folder = 'intertwined'

# Initialize the text splitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,  # striding over the text
    length_function=len,
)

# Initialize an empty list to store all text chunks
texts = []

# Function to extract and split text from a single PDF
def extract_and_split_text(pdf_file):
    doc_reader = PdfReader(pdf_file)
    raw_text = ''
    for i, page in enumerate(doc_reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    # Split the extracted text into chunks
    return text_splitter.split_text(raw_text)

# Loop over all files in the folder and process PDF files
for filename in os.listdir(pdf_folder):
    if filename.endswith('.pdf'):
        pdf_file = os.path.join(pdf_folder, filename)
        texts.extend(extract_and_split_text(pdf_file))

docs = []
for text in texts:
    temp = Document(
        page_content = text
    )
    vector_store.add_documents([temp])






print("Data preprocessed and saved successfully.")
