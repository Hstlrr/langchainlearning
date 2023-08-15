```py

import os

os.environ['OPENAI_API_KEY'] = "<OPENAI_API_KEY>"
os.environ['ACTIVELOOP_TOKEN'] = "<ACTIVELOOP_TOKEN>"


#The download_mp4_from_youtube() function will download the best quality mp4 video file from any YouTube link and save it to the specified path and filename

import yt_dlp

def download_mp4_from_youtube(url):
    # Set the options for the download
    filename = 'lecuninterview.mp4'
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
        'outtmpl': filename,
        'quiet': True,
    }

    # Download the video file
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=True)

url = "https://www.youtube.com/watch?v=mBjPyte2ZZo"
download_mp4_from_youtube(url)


import whisper

model = whisper.load_model("base")
result = model.transcribe("lecuninterview.mp4")
print(result['text'])

#load_model() method to download the model and transcribe a video file. Whisper has a variety of differentmodels like tiny, base small medium and large with tradeoffs for each

with open ('text.txt', 'w') as file:
    file.write(result['text']) #Saving to a txt file


```

# Summarisation with langchain alone

```py

#Import necessary libraries

from langchain import OpenAI, LLMChain
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

llm = OpenAI(model_name="text-davinci-003", temperature=0) #temp needs to be 0

# Use RecursiveTextSplitter to split input text into smaller chunks for the LLM

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"] #Using empty spaces and commas as sepaators to ensure efficient processing

from langchain.docstore.document import Document

with open('text.txt') as f:
    text = f.read()

texts = text_splitter.split_text(text) # the text_splitter is essentially RecursiveCharacterTextSplitter soo it'll have split_text as a method
docs = [Document(page_content=t) for t in texts[:4]]
# Each Document object is initialized with the content of a chunk from the texts list. The [:4] slice notation indicates that only the first four chunks will be used to create the Document objects.


from langchain.chains.summarize import load_summarize_chain
import textwrap

chain = load_summarize_chain(llm, chain_type="map_reduce")

output_summary = chain.run(docs)
wrapped_text = textwrap.fill(output_summary, width=100)
print(wrapped_text) # The textwrap library in Python provides a convenient way to wrap and format plain text by adjusting line breaks in an input paragraph

#Using the summarise_chain to create a summary

)

prompt_template = """Write a concise bullet point summary of the following:


{text}


CONSCISE SUMMARY IN BULLET POINTS:"""

BULLET_POINT_PROMPT = PromptTemplate(template=prompt_template,
                        input_variables=["text"])

chain = load_summarize_chain(llm,
                             chain_type="stuff",
                             prompt=BULLET_POINT_PROMPT)

output_summary = chain.run(docs)

wrapped_text = textwrap.fill(output_summary,
                             width=1000,
                             break_long_words=False,
                             replace_whitespace=False)
print(wrapped_text)
#The "stuff" approach is the simplest and most naive one, in which all the text from the transcribed video is used in a single prompt. This method may raise exceptions if all text is longer than the available context size of the LLM and may not be the most efficient way to handle large amounts of text.


#The 'refine' summarization chain is a method for generating more accurate and context-aware summaries. This chain type is designed to iteratively refine the summary by providing additional context when needed. That means: it generates the summary of the first chunk. Then, for each successive chunk, the work-in-progress summary is integrated with new info from the new chunk.
chain = load_summarize_chain(llm, chain_type="refine")

output_summary = chain.run(docs)
wrapped_text = textwrap.fill(output_summary, width=100)
print(wrapped_text)

```

# Adding Transcript to Deep Lake

```py

import yt_dlp

def download_mp4_from_youtube(urls, job_id):
    # This will hold the titles and authors of each downloaded video
    video_info = []

    for i, url in enumerate(urls):
        # Set the options for the download
        file_temp = f'./{job_id}_{i}.mp4'
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
            'outtmpl': file_temp,
            'quiet': True,
        }

        # Download the video file
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=True)
            title = result.get('title', "")
            author = result.get('uploader', "")

        # Add the title and author to our list
        video_info.append((file_temp, title, author))

    return video_info

urls=["https://www.youtube.com/watch?v=mBjPyte2ZZo&t=78s",
    "https://www.youtube.com/watch?v=cjs7QKJNVYM",]
vides_details = download_mp4_from_youtube(urls, 1)

# The script for video downloading has been modified to work with a list of URLs

import whisper

# load the model
model = whisper.load_model("base")

# iterate through each video and transcribe
results = []
for video in vides_details:
    result = model.transcribe(video[0])
    results.append( result['text'] )
    print(f"Transcription for {video[0]}:\n{result['text']}\n")

with open ('text.txt', 'w') as file:
    file.write(results['text']) # Whisper to transcribe video and save as text file



    from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load the texts
with open('text.txt') as f:
    text = f.read()
texts = text_splitter.split_text(text)

# Split the documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
    )
texts = text_splitter.split_text(text)

# Text splitter to get ready to load in as documents

from langchain.docstore.document import Document

docs = [Document(page_content=t) for t in texts[:4]]


#import Deep Lake and build a database with embedded documents:
from langchain.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

# create Deep Lake dataset
# TODO: use your organization id here. (by default, org id is your username)
my_activeloop_org_id = "<YOUR-ACTIVELOOP-ORG-ID>"
my_activeloop_dataset_name = "langchain_course_youtube_summarizer"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
db.add_documents(docs)


retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['k'] = 4
#The distance metric determines how the Retriever measures "distance" or similarity between different data points in the database. By setting distance_metric to 'cos', the Retriever will use cosine similarity as its distance metric

from langchain.prompts import PromptTemplate
prompt_template = """Use the following pieces of transcripts from a video to answer the question in bullet points and summarized. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Summarized answer in bullter points:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

from langchain.chains import RetrievalQA

chain_type_kwargs = {"prompt": PROMPT}
qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=retriever,
                                 chain_type_kwargs=chain_type_kwargs)

print( qa.run("Summarize the mentions of google according to their AI program") )

#Because we're using a RetrievalQA we need to use chain_type_kwargs

```
