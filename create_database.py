import os
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

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

# Create embeddings and initialize FAISS vector store
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(texts, embeddings)

# Save the text data using pickle
with open('preprocessed_texts.pkl', 'wb') as f:
    pickle.dump(texts, f, protocol=pickle.HIGHEST_PROTOCOL)

# Save the FAISS index separately
docsearch.save_local('faiss_index')

print("Data preprocessed and saved successfully.")
