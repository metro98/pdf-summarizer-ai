

import shutil

shutil.rmtree("db", ignore_errors=True)

import os




from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize LLM
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    temperature=0,
    model="gpt-3.5-turbo"
)

# Load PDF
loader = PyPDFLoader("example.pdf")
pages = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(pages)

# Create vector store
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vectordb = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="db"
)

# Create retrieval QA chain
retriever = vectordb.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# Summarize
summary = qa_chain.invoke("Summarize this document in 3 paragraphs.")

print("=== SUMMARY ===")
print(summary)

# Example question
question = "What are the challenges?"
answer = qa_chain.invoke(question)
print("\n=== ANSWER ===")
print(answer)
