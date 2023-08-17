#SCRAPE CONTENT

```py
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import re
# Load environment variables from the .env file
load_dotenv()
# Get the dataset path from the environment variable
dataset_path= os.environ.get('DEEPLAKE_DATASET_PATH')

embeddings =  OpenAIEmbeddings()

def get_documentation_urls():
    # List of relative URLs for Hugging Face documentation pages, commented a lot of these because it would take too long to scrape all of them
    return [
    '/docs/huggingface_hub/guides/overview',
    '/docs/huggingface_hub/guides/download',
    '/docs/huggingface_hub/guides/upload',
    '/docs/huggingface_hub/guides/hf_file_system',
    '/docs/huggingface_hub/guides/repository',
    '/docs/huggingface_hub/guides/search',
    # '/docs/huggingface_hub/guides/inference',
    # '/docs/huggingface_hub/guides/community',
    # '/docs/huggingface_hub/guides/manage-cache',
    # '/docs/huggingface_hub/guides/model-cards',
    # '/docs/huggingface_hub/guides/manage-spaces',
    # '/docs/huggingface_hub/guides/integrations',
    # '/docs/huggingface_hub/guides/webhooks_server',
    # Add the rest of the URLs here
    ]


def construct_full_url(base_url, relative_url):
    # Construct the full URL by appending the relative URL to the base URL
    return base_url + relative_url


def scrape_page_content(url):
    # Send a GET request to the URL and parse the HTML response using BeautifulSoup
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Extract the desired content from the page (in this case, the body text)
    text=soup.body.text.strip()
    # Remove non-ASCII characters
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\xff]', '', text)
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def scrape_all_content(base_url, relative_urls, filename):
    # Loop through the list of URLs, scrape content, and add it to the content list
    content = []
    for relative_url in relative_urls:
        full_url = construct_full_url(base_url, relative_url)
        scraped_content = scrape_page_content(full_url)
        content.append(scraped_content.rstrip('\n'))

    # Write the scraped content to a file
    with open(filename, 'w', encoding='utf-8') as file:
        for item in content:
            file.write("%s\n" % item)

    return content

# Define a function to load documents from a file
def load_docs(root_dir,filename):
    # Create an empty list to hold the documents
    docs = []
    try:
        # Load the file using the TextLoader class and UTF-8 encoding
        loader = TextLoader(os.path.join(
            root_dir, filename), encoding='utf-8')
        # Split the loaded file into separate documents and add them to the list of documents
        docs.extend(loader.load_and_split())
    except Exception as e:
        # If an error occurs during loading, ignore it and return an empty list of documents
        pass
    # Return the list of documents
    return docs


def split_docs(docs):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(docs)

def load_vectors_into_deeplake(dataset_path, source_chunks):
    # Initialize the DeepLake database with the dataset path and embedding function
    deeplake_db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
    # Add the text chunks to the database
    deeplakedb=deeplake_db.add_texts(source_chunks)
    return deeplakedb


# Define the main function
def main():
    base_url = 'https://huggingface.co'
    # Set the name of the file to which the scraped content will be saved
    filename='content.txt'
    # Set the root directory where the content file will be saved
    root_dir ='./'
    relative_urls = get_documentation_urls()
    # Scrape all the content from the relative urls and save it to the content file
    content = scrape_all_content(base_url, relative_urls,filename)
    # Load the content from the file
    docs = load_docs(root_dir,filename)
    # Split the content into individual documents
    texts = split_docs(docs)
    # Create a DeepLake database with the given dataset path and embedding function
    db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
    # Add the individual documents to the database
    db.add_documents(texts)
    # Clean up by deleting the content file
    os.remove(filename)

# Call the main function if this script is being run as the main program
if __name__ == '__main__':
    main()


```

#MAIN

```py
import os
import openai
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from elevenlabs import generate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from streamlit_chat import message
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Constants
TEMP_AUDIO_PATH = "temp_audio.wav"
AUDIO_FORMAT = "audio/wav"

# Load environment variables from .env file and return the keys
openai.api_key = os.environ.get('OPENAI_API_KEY')
eleven_api_key = os.environ.get('ELEVEN_API_KEY')
active_loop_data_set_path = os.environ.get('DEEPLAKE_DATASET_PATH')

# Load embeddings and DeepLake database
def load_embeddings_and_database(active_loop_data_set_path):
    embeddings = OpenAIEmbeddings()
    db = DeepLake(
        dataset_path=active_loop_data_set_path,
        read_only=True,
        embedding_function=embeddings
    )
    return db

# Transcribe audio using OpenAI Whisper API
def transcribe_audio(audio_file_path, openai_key):
    openai.api_key = openai_key
    try:
        with open(audio_file_path, "rb") as audio_file:
            response = openai.Audio.transcribe("whisper-1", audio_file)
        return response["text"]
    except Exception as e:
        print(f"Error calling Whisper API: {str(e)}")
        return None

# Record audio using audio_recorder and transcribe using transcribe_audio
def record_and_transcribe_audio():
    audio_bytes = audio_recorder()
    transcription = None
    if audio_bytes:
        st.audio(audio_bytes, format=AUDIO_FORMAT)

        with open(TEMP_AUDIO_PATH, "wb") as f:
            f.write(audio_bytes)

        if st.button("Transcribe"):
            transcription = transcribe_audio(TEMP_AUDIO_PATH, openai.api_key)
            os.remove(TEMP_AUDIO_PATH)
            display_transcription(transcription)

    return transcription

# Display the transcription of the audio on the app
def display_transcription(transcription):
    if transcription:
        st.write(f"Transcription: {transcription}")
        with open("audio_transcription.txt", "w+") as f:
            f.write(transcription)
    else:
        st.write("Error transcribing audio.")

# Get user input from Streamlit text input field
def get_user_input(transcription):
    return st.text_input("", value=transcription if transcription else "", key="input")

# Search the database for a response based on the user's query
def search_db(user_input, db):
    print(user_input)
    retriever = db.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['fetch_k'] = 100
    retriever.search_kwargs['maximal_marginal_relevance'] = True
    retriever.search_kwargs['k'] = 10
    model = ChatOpenAI(model='gpt-3.5-turbo')
    qa = RetrievalQA.from_llm(model, retriever=retriever, return_source_documents=True)
    return qa({'query': user_input})

# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i],key=str(i))
        #Voice using Eleven API
        voice= "Bella"
        text= history["generated"][i]
        audio = generate(text=text, voice=voice,api_key=eleven_api_key)
        st.audio(audio, format='audio/mp3')

# Main function to run the app
def main():
    # Initialize Streamlit app with a title
    st.write("# JarvisBase 🧙")

    # Load embeddings and the DeepLake database
    db = load_embeddings_and_database(active_loop_data_set_path)

    # Record and transcribe audio
    transcription = record_and_transcribe_audio()

    # Get user input from text input or audio transcription
    user_input = get_user_input(transcription)

    # Initialize session state for generated responses and past messages
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I am ready to help you"]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey there!"]

    # Search the database for a response based on user input and update session state
    if user_input:
        output = search_db(user_input, db)
        print(output['source_documents'])
        st.session_state.past.append(user_input)
        response = str(output["result"])
        st.session_state.generated.append(response)

    # Display conversation history using Streamlit messages
    if st.session_state["generated"]:
        display_conversation(st.session_state)

# Run the main function when the script is executed
if __name__ == "__main__":
    main()

```
