import os
from getpass import getpass
from llama_index import SimpleDirectoryReader
from llama_index import ServiceContext
from llama_index import VectorStoreIndex
from llama_index import Document
from llama_index.llms import OpenAI
from pydub import AudioSegment as am
import random
from llama_index.embeddings import OpenAIEmbedding, GooglePaLMEmbedding
from Advanced_RAG_Techniques import get_sentence_window_query_engine, build_sentence_window_index
import dotenv
import tempfile
import streamlit as st
import speech_recognition
from speech_recognition import Recognizer, Microphone
from googletrans import Translator
import audio_recorder_streamlit as ars
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass("Provide your Google API key here")

r = Recognizer()

SUPPORTED_LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Punjabi": "pa",
    "Marathi": "mr",
    "Bengali": "bn",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
}

# Initialize translator
translator = Translator()

# Streamlit interface
st.title("Advanced RAG with Google PaLM Embeddings")
st.write("This application allows you to ask questions about the document through text or voice, and select your preferred language.")

# Document loading
uploaded_files = st.file_uploader("Upload your documents (PDFs)", accept_multiple_files=True)

# Language selection
with st.sidebar:
    source_language = st.selectbox("Choose your preferred language", list(SUPPORTED_LANGUAGES.keys()))
    target_language = st.selectbox("Choose the Language your Document is in", list(SUPPORTED_LANGUAGES.keys()))
    #microphone = st.button("🎤 Speak your question")

# Process document and build index
if uploaded_files:
    doc_texts = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name
            documents = SimpleDirectoryReader(input_files=[temp_path])
            documents = documents.load_data()
            doc_texts.append("\n\n".join([doc.text for doc in documents]))

    doc = Document(text="\n\n".join([doc.text for doc in documents]))

    llm = llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        verbose=True,
        temperature=0.1,
        max_tokens=4096,
        google_api_key="AIzaSyB1JNQU4vjOvUgN3XBqJFjDqJczgY9oQKc"
    )
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    try:
        sentence_index = build_sentence_window_index(doc, llm, embed_model=embedding_model,
                                                      save_dir=f"{uploaded_file.name.split('.')[0]}")
        sentence_window_engine = get_sentence_window_query_engine(sentence_index)
    except Exception as e:
        st.error(f"Error building document index: {e}")

    # Process user query
    mic = st.checkbox("Check to enable the Microphone")
    if mic:
      microphone = ars.audio_recorder(text = "Click to record", icon_size="2x", key="audio_button")
      if microphone:
            filename = str(random.randint(1,199))+".wav"
            with open(filename, mode='bx') as f:
                f.write(microphone)
                sound = am.from_file(filename, format='wav', frame_rate=44100)
                sound = sound.set_frame_rate(16000)
                sound.export(filename, format='wav')
                harvard = speech_recognition.AudioFile(filename)
                with harvard as source:
                    audio = r.record(source)
            try:
                recognized_text = r.recognize_google(audio, language=source_language)
                user_query = recognized_text
                translated_query = translator.translate(user_query, src=source_language, dest=target_language).text
            except Exception as e:
                st.error(f"Error recognizing speech: {e}")

            if translated_query:
                  try:
                      # Get response from the document
                      response = sentence_window_engine.query(translated_query)

                      # Extract text content from the response
                      translated_response = str(response)

                      # Translate the extracted text
                      translated_response = translator.translate(translated_response, src=target_language,
                                                                dest=source_language).text

                      # Display the translated response
                      st.write(f"Shri Krishna says: {translated_response}")
                  except Exception as e:
                      st.error(f"Error processing query: {e}")

    else:
        user_query = st.text_input("Enter your question:")

    # Check for empty user query
        if not user_query:
            st.warning("Please enter a question or speak into the microphone.")
        else:
            # Translate user query to document language
            try:
                translated_query = translator.translate(user_query, src=source_language, dest=target_language).text
            except Exception as e:
                st.error(f"Error translating query: {e}")
                translated_query = None

            # Process the query if translation is successful
            if translated_query:
                try:
                    # Get response from the document
                    response = sentence_window_engine.query(translated_query)

                    # Extract text content from the response
                    translated_response = str(response)

                    # Translate the extracted text
                    translated_response = translator.translate(translated_response, src=target_language,
                                                              dest=source_language).text

                    # Display the translated response
                    st.write(f"Response: {translated_response}")
                except Exception as e:
                    st.error(f"Error processing query: {e}")


