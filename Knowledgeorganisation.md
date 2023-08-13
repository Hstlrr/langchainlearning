- An index is a data structure thatr organises and stores documents to enable efficient searching
- A retriver harnesses the index to locate and return the relevent documents based on user queries
- Vector databases being used for primarily for index types

```py
from langchain.document_loaders import TextLoader

# text to write to a local file
# taken from https://www.theverge.com/2023/3/14/23639313/google-ai-language-model-palm-api-challenge-openai
text = """Google opens up its AI language model PaLM to challenge OpenAI and GPT-3
Google is offering developers access to one of its most advanced AI language models: PaLM.
The search giant is launching an API for PaLM alongside a number of AI enterprise tools
it says will help businesses “generate text, images, code, videos, audio, and more from
simple natural language prompts.”

PaLM is a large language model, or LLM, similar to the GPT series created by OpenAI or
Meta’s LLaMA family of models. Google first announced PaLM in April 2022. Like other LLMs,
PaLM is a flexible system that can potentially carry out all sorts of text generation and
editing tasks. You could train PaLM to be a conversational chatbot like ChatGPT, for
example, or you could use it for tasks like summarizing text or even writing code.
(It’s similar to features Google also announced today for its Workspace apps like Google
Docs and Gmail.)
"""

# write text to local file
with open("my_file.txt", "w") as file:
    file.write(text)

# use TextLoader to load text from local file
loader = TextLoader("my_file.txt")
docs_from_file = loader.load()



from langchain.text_splitter import CharacterTextSplitter

# create a text splitter
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)

# split documents into chunks
docs = text_splitter.split_documents(docs_from_file)

print(len(docs))
# 2

from langchain.embeddings import OpenAIEmbeddings

# Before executing the following code, make sure to have
# your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

```

# Using Deep Lake for datastoring

```py
from langchain.vectorstores import DeepLake

# Before executing the following code, make sure to have your
# Activeloop key saved in the “ACTIVELOOP_TOKEN” environment variable.

# create Deep Lake dataset
# TODO: use your organization id here. (by default, org id is your username)
my_activeloop_org_id = "<YOUR-ACTIVELOOP-ORG-ID>"
my_activeloop_dataset_name = "langchain_course_indexers_retrievers"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# add documents to our Deep Lake dataset
db.add_documents(docs)


# create retriever from db
retriever = db.as_retriever()


from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# create a retrieval chain
qa_chain = RetrievalQA.from_chain_type(
	llm=OpenAI(model="text-davinci-003"),
	chain_type="stuff",
	retriever=retriever
)

query = "How Google plans to challenge OpenAI?"
response = qa_chain.run(query)
print(response)

```

# Using a Compressor to ensure only relevant parts of docs returned given a specific query

```py
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# create GPT3 wrapper
llm = OpenAI(model="text-davinci-003", temperature=0)

# create compressor for the retriever
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
	base_compressor=compressor,
	base_retriever=retriever
)

# retrieving compressed documents
retrieved_docs = compression_retriever.get_relevant_documents(
	"How Google plans to challenge OpenAI?"
)
print(retrieved_docs[0].page_content)


```

# Different Types of loaders

# TEXT LOADER

Text Loader to load and handle plain text files

```py
from langchain.document_loaders import TextLoader
loader = TextLoader('file_path.txt')
documents = loader.load()

```

#PyPDFLoader for PDF Files

```py
!pip install -q pypdf

from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("example_data/layout-parser-paper.pdf")
pages = loader.load_and_split()

print(pages[0])

```

#SeleniumURLLoader to handle HTML Documents i.e. websites that require JS rendering

```py

from langchain.document_loaders import SeleniumURLLoader

urls = [
    "https://www.youtube.com/watch?v=TFa539R09EQ&t=139s",
    "https://www.youtube.com/watch?v=6Zv6A_9urh4&t=112s"
]

loader = SeleniumURLLoader(urls=urls)
data = loader.load()

print(data[0])


```

The SeleniumURLLoader class includes the following attributes:

URLs (List[str]): List of URLs to load.
continue_on_failure (bool, default=True): Continues loading other URLs on failure if True.
browser (str, default="chrome"): Browser selection, either 'Chrome' or 'Firefox'.
executable_path (Optional[str], default=None): Browser executable path.
headless (bool, default=True): Browser runs in headless mode if True.

```py
loader = SeleniumURLLoader(urls=urls, browser="firefox") # Can customise the attributes above


```

# GDrive Loader

Requires GCloud and a Google Drive API enabled on service account
Will need to create a key

```py
from langchain.document_loaders import GoogleDriveLoader
loader = GoogleDriveLoader(
    folder_id="your_folder_id",
    recursive=False  # Optional: Fetch files from subfolders recursively. Defaults to False.
)

docs = loader.load()
```

# What are Text Splitters and their use

LLMs have a maximum prompt size, preventing them from feeding entire documents.

This makes it crucial to divide documents into smaller parts, and Text Splitters prove to be extremely useful in achieving this.

Text Splitters help break down large text documents into smaller, more digestible pieces that language models can process more effectively.

Using a Text Splitter can also improve vector store search results, as smaller segments might be more likely to match a query. Experimenting with different chunk sizes and overlaps can be beneficial in tailoring results to suit your specific needs.

# Character text splitter

Basic splitter using chunks and overlap

```py
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("The One Page Linux Manual.pdf")
pages = loader.load_and_split()

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
texts = text_splitter.split_documents(pages)

print(texts[0])

print (f"You have {len(texts)} documents")

print ("Preview:")
print (texts[0].page_content)

```

# Recursive Text Splitter

The Recursive Character Text Splitter is a text splitter designed to split the text into chunks based on a list of characters provided.

It attempts to split text using the characters from a list in order until the resulting chunks are small enough. By default, the list of characters used for splitting is ["\n\n", "\n", " ", "], which tries to keep paragraphs, sentences, and words together as long as possible, as they are generally the most semantically related pieces of text.

```Py

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load a long document
with open('The One Page Linux Manual.pdf', encoding= 'unicode_escape') as f:
    sample_text = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    length_function=len, # Gives count of numbers of charc in a chunk
)

texts = text_splitter.create_documents([sample_text])
print(texts)

```

# NLTK Splitter

The NLTKTextSplitter in LangChain is an implementation of a text splitter that uses the Natural Language Toolkit (NLTK) library to split text based on tokenizers. The goal is to split long texts into smaller chunks without breaking the structure of sentences and paragraphs.

```py
# Load a long document
with open('/home/cloudsuperadmin/scrape-chain/langchain/LLM.txt', encoding= 'unicode_escape') as f:
    sample_text = f.read()

from langchain.text_splitter import NLTKTextSplitter
text_splitter = NLTKTextSplitter(chunk_size=500)


texts = text_splitter.split_text(sample_text)
print(texts)

```

# SpacyTextSplitter

The SpacyTextSplitter helps split large text documents into smaller chunks based on a specified size

You can create a SpacyTextSplitter object by specifying the chunk_size parameter, measured by a length function passed to it, which defaults to the number of characters

```py

from langchain.text_splitter import SpacyTextSplitter


# Load a long document
with open('/home/cloudsuperadmin/scrape-chain/langchain/LLM.txt', encoding= 'unicode_escape') as f:
    sample_text = f.read()

# Instantiate the SpacyTextSplitter with the desired chunk size
text_splitter = SpacyTextSplitter(chunk_size=500, chunk_overlap=20)


# Split the text using SpacyTextSplitter
texts = text_splitter.split_text(sample_text)

# Print the first chunk
print(texts[0])

```

#TokenTextSplitter

The main advantage of using TokenTextSplitter over other text splitters, like CharacterTextSplitter, is that it respects the token boundaries, ensuring that the chunks do not split tokens in the middle. This can be particularly helpful in maintaining the semantic integrity of the text when working with language models and embeddings

```py
from langchain.text_splitter import TokenTextSplitter

# Load a long document
with open('/home/cloudsuperadmin/scrape-chain/langchain/LLM.txt', encoding= 'unicode_escape') as f:
    sample_text = f.read()

# Initialize the TokenTextSplitter with desired chunk size and overlap
text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=50)

# Split into smaller chunks
texts = text_splitter.split_text(sample_text)
print(texts[0])


```
