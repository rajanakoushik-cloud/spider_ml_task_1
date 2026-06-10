import streamlit as st
import os
from pypdf import PdfReader
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

token = "tokens"
if not token:
    st.error("Missing Hugging Face Token! Make sure your .env file contains tokens=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")
    st.stop()

st.title("AI Assistant CHITTI")

pdf_files = [
    "1706.03762v7.pdf",
    "1810.04805v2.pdf",
    "1908.10084v1.pdf",
    "2005.11401v4.pdf",
    "2005.14165v4.pdf",
    "2106.09685v2.pdf",
    "2307.09288v2.pdf"
]

all_chunks = []

for file_name in pdf_files:
    if os.path.exists(file_name):
        reader = PdfReader(file_name)
        for page_number, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text:
                continue
            
            words = text.split()
            for i in range(0, len(words), 150):
                chunk_words = words[i:i+200]
                chunk_text = " ".join(chunk_words)
                
                if len(chunk_text.strip()) > 10:
                    all_chunks.append({
                        "text": chunk_text,
                        "source": file_name,
                        "page": page_number + 1
                    })

st.success(f"Successfully loaded {len(all_chunks)} text segments from your PDFs!")

client = InferenceClient(token=token)

user_query = st.text_input("Ask a question about the papers:")

if st.button("Submit Question"):
    if user_query:
        with st.spinner("Searching text and generating answer..."):
            
            query_words = user_query.lower().split()
            scored_chunks = []
            
            # FIXED: Corrected indentation, variable definitions, and syntax brackets
            for chunk in all_chunks:
                score = 0
                chunk_text_lower = chunk["text"].lower()
                for word in query_words:
                    if len(word) > 3 and word in chunk_text_lower:
                        score += 1
                if score > 0:
                    scored_chunks.append((score, chunk))
            
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            top_matches = scored_chunks[:4]
            
            if not top_matches:
                top_matches = [(0, c) for c in all_chunks[:4]]
            
            context_string = ""
            retrieved_sources = []
            for score, chunk in top_matches:
                doc_info = f"\nSource File: {chunk['source']}, Page: {chunk['page']}\nText: {chunk['text']}\n"
                context_string = context_string + doc_info
                retrieved_sources.append(chunk)
                
            system_instructions = (
                "You are an AI assistant that answers questions using only the provided context data. "
                "You must explicitly state which file and page number your facts came from. "
                "If the context data does not contain the answer, say 'I cannot find the answer based on the papers.' "
                "Do not make up facts outside the text data."
            )
            
            try:
                api_response = client.chat.completions.create(
                    model="Qwen/Qwen2.5-7B-Instruct",
                    messages=[
                        {"role": "system", "content": system_instructions},
                        {"role": "user", "content": f"Context data:\n{context_string}\n\nUser Question: {user_query}"}
                    ],
                    max_tokens=500,
                    temperature=0.1
                )
                
                answer_text = api_response.choices[0].message.content
                
                st.subheader("Answer:")
                st.write(answer_text)
                
                st.subheader("Sources used:")
                for i, src in enumerate(retrieved_sources):
                    st.write(f"**Source {i+1}:** {src['source']} (Page {src['page']})")
                    st.text(src['text'])
            except Exception as e:
                st.error(f"Hugging Face API Error: {e}")
