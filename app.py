import os
from pathlib import Path

import gradio as gr
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers.string import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# -------- Configuration --------
# Set your OpenAI API key via environment variable or .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not set. Set it before running for best results.")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

DATA_FILE = Path("data/2024_state_of_the_union.txt")

# -------- Load & Split Documents --------
if not DATA_FILE.exists():
    raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    raw_text = f.read()

text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

docs = text_splitter.create_documents([raw_text])

# -------- Embeddings & Vector Store --------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma(
    collection_name="rag_chatbot_sotu_2024",
    embedding_function=embeddings
)

# Load documents only if collection is empty (idempotent run)
if len(vector_store.get()["ids"]) == 0:
    vector_store.add_documents(docs)

retriever = vector_store.as_retriever()

# -------- LLM & Prompt --------
llm = ChatOpenAI(model="gpt-4o-mini")  # choose a small, inexpensive model

prompt_template = PromptTemplate(
    template=(
        "Use the context to answer the question.
"
        "If you don't know the answer, say you don't know.

"
        "context:
{context}

"
        "question: {question}

"
        "answer:"
    )
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

# -------- Gradio Interface --------
def chat_fn(message, history):
    answer = rag_chain.invoke(message)
    history = history or []
    history.append((message, answer))
    return history, history

with gr.Blocks() as demo:
    gr.Markdown("# RAG Chatbot — State of the Union 2024")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask something from the transcript...")
    clear = gr.Button("Clear")
    msg.submit(chat_fn, [msg, chatbot], [chatbot, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()
