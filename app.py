import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI 
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv 
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import os 

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
qdrant_client = QdrantClient(
    url="https://66db3bed-f1a5-4513-bf57-a214be870f4c.eu-west-2-0.aws.cloud.qdrant.io:6333", 
    api_key="KbN6Upf3ixu-RR7FlXLu6m9RtesvGG_EHjMZ780zrq9bVJDa014ehQ",
)

def create_collection_if_not_exists():
    collections = qdrant_client.get_collections()  
    collection_name = "collection" 

    if collection_name not in collections:
        
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)  
        )
        print(f"Collection '{collection_name}' created successfully.")
    else:
        print(f"Collection '{collection_name}' already exists.")


# Read the pdfs and extract details from the pages
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Converts text to chunks 
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

# Conversational chain for Q&A
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, and make sure to provide all the details. If the answer is not in 
    the provided context, just say, "answer is not available in the context." Don't provide the wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n {question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# User input for Q&A
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Search for relevant documents
    search_results = qdrant_client.search(
        collection_name="collection",  
        query_vector=embeddings.embed_query(user_question),
        limit=5
    )
    
    docs = [result.payload["text"] for result in search_results]

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    st.write("Reply: ", response.get("output_text", "No response generated"))

def main():
    st.set_page_config(page_title="Chat with Multiple PDFs")
    st.header("Chat with Multiple PDF using Gemini")

    user_question = st.text_input("Ask a question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit and Process",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                st.success("Done")

if __name__ == "__main__":
    main()
