from langchain.document_loaders import WikipediaLoader

from langchain.vectorstores import Qdrant

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.llms import GPT4All

from langchain.embeddings import HuggingFaceEmbeddings

def init_embeddings():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    return embeddings

def init_llm():
    llm = GPT4All(model="D:/DProgFiles/GPT4All/models/ggml-model-gpt4all-falcon-q4_0.bin", max_tokens=2048)
    return llm

def init_text_splitter():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=2)
    return text_splitter