from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from operator import itemgetter
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


def get_llm(api_key):
    return ChatOpenAI(
        temperature=0.1,
        streaming=True,
        openai_api_key=api_key,
        callbacks=[
            ChatCallbackHandler(),
        ],
    )


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file, api_key):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

def get_memory():
    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationBufferMemory(return_messages=True)
    return st.session_state["memory"]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context and conversation history. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
"""
)

with st.sidebar:
    st.markdown(
        """
        Source Code:
        [![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?style=for-the-badge&logo=github)](https://github.com/nomadcoders/nc-gpt-challenge-5)
        ---
        """
    )
    
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key to use the chatbot"
    )
    
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

if file and api_key:
    # 파일이나 API 키가 바뀌면 메모리 초기화
    current_file_key = f"{file.name}_{hash(api_key)}"
    if "current_file_key" not in st.session_state or st.session_state["current_file_key"] != current_file_key:
        st.session_state["current_file_key"] = current_file_key
        st.session_state["memory"] = ConversationBufferMemory(return_messages=True)
        st.session_state["messages"] = []
    
    retriever = embed_file(file, api_key)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        llm = get_llm(api_key)
        chain = (
            { 
                "context": retriever | RunnableLambda(format_docs), 
                "question": RunnablePassthrough(),
                "history": RunnableLambda(
                    st.session_state["memory"].load_memory_variables
                ) | itemgetter("history"),
            } 
            | prompt 
            | llm
        )
        with st.chat_message("ai"):
            result = chain.invoke(message)
            memory = get_memory()
            memory.save_context(
                {"input": message},
                {"output": result.content},
            )
elif file and not api_key:
    st.error("🔑 Please enter your OpenAI API key in the sidebar to start embedding and using the chatbot.")
elif not file and api_key:
    st.info("📄 Please upload a file in the sidebar to start using the chatbot.")
else:
    st.info("🔑📄 Please enter your OpenAI API key and upload a file in the sidebar to start using the chatbot.")
    st.session_state["messages"] = []
    st.session_state["memory"] = ConversationBufferMemory(return_messages=True)