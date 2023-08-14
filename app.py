import gradio as gr

from utils import init_embeddings, init_llm, init_text_splitter
from utils import build_knowledge, chat

from langchain.document_loaders import WikipediaLoader

from langchain.vectorstores import Qdrant

from langchain.chains import RetrievalQA
from langchain import PromptTemplate


embeddings = init_embeddings()
llm = init_llm()
text_splitter = init_text_splitter()

docs = None
qdrant = None

def build_knowledge(query_topic, max_docs):
    global docs, qdrant
    docs = WikipediaLoader(query=query_topic, load_max_docs=max_docs).load()
    docs = text_splitter.split_documents(docs)

    qdrant = Qdrant.from_documents(
        docs,
        embeddings,
        location=":memory:",
        collection_name="wikitest"
    )

    return "built"

def chat(question):
    global qdrant, docs
    if qdrant is None or docs is None:
        "Please enter topic first to build knowledge base"

    context = qdrant.similarity_search(question)

    template = """Use the following pieces of context to answer the question at the end. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=qdrant.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    print(context)
    result = qa_chain({'query': question})

    return result['result']


with gr.Blocks() as demo:
    query_topic = gr.Textbox(label="I want to know about...", placeholder="Messi")
    max_docs = gr.Number(default=2, label="Max docs", precision=0)
    
    built = gr.Textbox()

    build_button = gr.Button(value="Build knowledge")
    build_button.click(fn=build_knowledge, inputs=[query_topic, max_docs], outputs=[built])


    question = gr.Textbox(label="Question")
    response = gr.Textbox(label="Response")

    btn = gr.Button("Ask")
    btn.click(fn=chat, inputs=[question], outputs=[response])

demo.launch(share=True)