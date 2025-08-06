

!pip install langchain openai chromadb tiktoken pypdf python-dotenv

from google.colab import files
uploaded = files.upload()

pip install sentence-transformers

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os

# Load PDF
loader = PyPDFLoader("Abhinav CV.pdf") #u have to chnage the name of the file here according to upload
docs = loader.load()

# Split text
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Use HuggingFace embeddings (FREE)
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(chunks, embedding)

# Use OpenRouter for LLM
os.environ["OPENAI_API_KEY"] = "Enter your API key here...."
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

llm = ChatOpenAI(model="mistralai/devstral-small-2505:free", temperature=0.7)

# Setup QA
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

# Ask a question
query = "who is abhinav sharma?"
print("Answer:", qa.run(query))