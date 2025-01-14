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

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url="https://66db3bed-f1a5-4513-bf57-a214be870f4c.eu-west-2-0.aws.cloud.qdrant.io:6333",
    api_key="KbN6Upf3ixu-RR7FlXLu6m9RtesvGG_EHjMZ780zrq9bVJDa014ehQ",
    timeout=60
)

def create_collection_if_not_exists():
    collection_name = "collection"
    collections = [col.name for col in qdrant_client.get_collections().collections]

    if collection_name not in collections:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        print(f"Collection '{collection_name}' created successfully.")
    else:
        print(f"Collection '{collection_name}' already exists.")

# Read the PDFs and extract details
def get_pdf_text(pdf_paths):
    text = ""
    for pdf_path in pdf_paths:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Converts text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

# Conversational chain for Q&A
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, 
    just say, "The answer is not available in the context." Do not provide incorrect answers.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# User input for Q&A
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Search for relevant documents in the Qdrant collection
    search_results = qdrant_client.search(
        collection_name="collection",
        query_vector=embeddings.embed_query(user_question),
        limit=5
    )
    docs = [result.payload["text"] for result in search_results]

    if not docs:
        st.write("No relevant documents found.")
        return

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.write("Reply:", response.get("output_text", "No response generated."))

def main():
    st.set_page_config(page_title="Chat with Multiple PDFs")
    st.header("Chat with Multiple PDFs using Gemini!!")

    # Predefined paths to PDF files
    pdf_paths = [
        "Data_Communication_Overview.pdf",
        "Digital_Signal_Processing.pdf"
    ]

    # Button to process PDFs
    if st.button("Process PDFs"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_paths)
            if not raw_text:
                st.error("Failed to extract text from the PDFs. Please check the files.")
                return

            text_chunks = get_text_chunks(raw_text)

            # Embed text chunks and upload to Qdrant
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectors = embeddings.embed_documents(text_chunks)
            print("Vector dimensions:", len(vectors[0]))
            payloads = [{"text": chunk} for chunk in text_chunks]

            create_collection_if_not_exists()
            qdrant_client.upload_collection(
                collection_name="collection",
                vectors=vectors,
                payload=payloads,
                batch_size=64
            )
            st.success("PDF data processed and uploaded to Qdrant successfully.")

    # Question input
    user_question = st.text_input("Ask a question based on the PDF content:")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
