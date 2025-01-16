import streamlit as st
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import retriever
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from dotenv import load_dotenv
import qdrant_client
import os
import pdfplumber

load_dotenv()


os.environ['QDRANT_HOST'] = ""
os.environ['QDRANT_API_KEY'] = ""

client = qdrant_client.QdrantClient(
    os.getenv('QDRANT_HOST'),
    api_key=os.getenv('QDRANT_API_KEY'),
    timeout=30

)

# Create collection
os.environ['QDRANT_COLLECTION_NAME'] = "My_Collection"

vectors_config = qdrant_client.http.models.VectorParams(
    size=1536,  # OpenAI
    distance=qdrant_client.http.models.Distance.COSINE
)

client.recreate_collection(
    collection_name=os.getenv('QDRANT_COLLECTION_NAME'),
    vectors_config=vectors_config
)

OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")



embeddings = OpenAIEmbeddings()

vector_store = Qdrant(
    client=client,
    collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
    embeddings=embeddings
)

def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Read the PDF document instead of the text file
pdf_paths = [
    "Digital_Signal_Processing Doc.pdf",
    "Data Communication.pdf"
    ]

all_texts = []  # This will hold all the text from all PDFs
for pdf_path in pdf_paths:
    raw_text = extract_text_from_pdf(pdf_path)
    texts = get_chunks(raw_text)
    all_texts.extend(texts)  # Add chunks from this PDF to the list

# Add all texts from all PDFs into the vector store
vector_store.add_texts(all_texts)


qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

# Streamlit interface
st.title("Q&A Chatbot")
st.write("Ask questions from the PDF Documents")


# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Enter your question:", "")
if st.button("Submit"):
    if query:
        response = qa.run(query)
        st.session_state.history.append((query, response))

if st.session_state.history:
    st.write("### Chat History")
    for q, r in st.session_state.history:
        st.write(f"**Question:** {q}")
        st.write(f"**Answer:** {r}")
