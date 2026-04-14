import streamlit as st
import pandas as pd
import folium
import torch
import re
import os
import sys
from streamlit_folium import st_folium

# Import LangChain and Transformers components
try:
    import langchain_community
    import langchain_core
except ImportError:
    st.error("Missing required libraries. Please install the dependencies in requirements.txt")

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

# =================================================================
# 📍 CONFIGURATION & PATHS
# NOTE: Update these paths relative to your repository root
# =================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "data", "austin_chroma_db") # Path to Vector DB
GEO_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "sample_listings.csv") # Path to CSV
LLM_MODEL_ID = "path/to/your/local/llm-model" # TODO: Update with your local model path

st.set_page_config(layout="wide", page_title="HousiSense: AI Housing Agent")

# =================================================================
# 🛠️ RESOURCE LOADING
# =================================================================
@st.cache_resource
def load_resources():
    """Load spatial data and initialize the RAG pipeline."""
    # 1. Load Geospatial Data
    if not os.path.exists(GEO_DATA_PATH):
        st.warning(f"Data file not found at {GEO_DATA_PATH}. Using fallback or empty dataframe.")
        df = pd.DataFrame()
    else:
        df = pd.read_csv(GEO_DATA_PATH)
        if 'id' in df.columns:
            df['id'] = df['id'].astype(str)

    # 2. Initialize Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )

    # 3. Load Vector Store
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # 4. Initialize LLM Pipeline
    # Using 4-bit quantization for efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.15
        )
        llm = HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"Failed to load LLM: {e}. Ensure the model path is correct.")
        llm = None

    return df, retriever, llm

df, retriever, llm = load_resources()

# =================================================================
# 🗺️ HELPER FUNCTIONS
# =================================================================
def format_docs_with_id(docs):
    """Format retrieved documents with their respective Listing IDs."""
    formatted = []
    for doc in docs:
        listing_id = doc.metadata.get('listing_id', 'Unknown')
        formatted.append(f"[Listing ID: {listing_id}]\n{doc.page_content}")
    return "\n\n".join(formatted)

def create_map(listing_id):
    """Generate a folium map centered on the recommended property."""
    if df.empty or listing_id not in df['id'].values:
        return folium.Map(location=[30.2672, -97.7431], zoom_start=12) # Default Austin center

    row = df[df['id'] == listing_id].iloc[0]
    m = folium.Map(location=[row['lat'], row['lon']], zoom_start=15)
    folium.Marker(
        [row['lat'], row['lon']], 
        popup=f"Listing {listing_id}",
        icon=folium.Icon(color='red', icon='home')
    ).add_to(m)
    # Add a 500m walking buffer
    folium.Circle([row['lat'], row['lon']], radius=500, color='blue', fill=True, fill_opacity=0.1).add_to(m)
    return m

# =================================================================
# 💻 UI LAYOUT (Streamlit)
# =================================================================
st.title("🏠 HousiSense: Cognitive Urban Housing Agent")
st.markdown("---")

col1, col2 = st.columns([1, 1])

if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_map" not in st.session_state:
    st.session_state.current_map = folium.Map(location=[30.2672, -97.7431], zoom_start=12)

with col1:
    st.subheader("💬 AI Reasoning Chat")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ex: I need a quiet place near UT Austin with good park access."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if llm is None:
                st.error("LLM not initialized. Check model path.")
            else:
                with st.spinner("Reasoning through spatial data..."):
                    # Prompt engineering for grounded spatial reasoning
                    template = """<|system|>
You are a spatial-cognitive AI agent called HousiSense. Use the provided context to recommend a house.
Always include the recommended 'Listing ID' in this exact format: "Listing ID: <number>".
Ground your answer strictly in the provided spatial narratives and guest reviews.
</s>
<|user|>
Context:
{context}

Question: 
{question}
</s>
<|assistant|>"""
                    prompt_template = PromptTemplate.from_template(template)
                    chain = (
                        {"context": retriever | format_docs_with_id, "question": RunnablePassthrough()}
                        | prompt_template
                        | llm
                        | StrOutputParser()
                    )
                    
                    try:
                        response = chain.invoke(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

                        # Extract Listing ID to trigger map update
                        match = re.search(r"Listing ID:?\s*(\d+)", response)
                        if match:
                            found_id = match.group(1)
                            st.session_state.current_map = create_map(found_id)
                            st.success(f"📍 Map updated for Listing ID: {found_id}")
                    except Exception as e:
                        st.error(f"Error during chain invocation: {e}")

with col2:
    st.subheader("🗺️ Location Map")
    st_folium(
        st.session_state.current_map, 
        width="100%", 
        height=600,
        key="main_map"
    )
