#import os
#from dotenv import load_dotenv

#load_dotenv()
import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS  # ✅ Fix: import FAISS

# Support both "langchain_classic" and regular "langchain"
try:
    from langchain_classic.chains import ConversationalRetrievalChain
    from langchain_classic.memory import ConversationBufferMemory
except ImportError:
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory


def get_pdf_text(pdf_docs):
    text_parts = []
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return splitter.split_text(text)


def get_embeddings():
    return OpenAIEmbeddings()


def get_conversational_chain(vectorstore):
    llm = ChatOpenAI(temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )


def handle_userinput(user_question):
    if st.session_state.conversation_chain is None:
        st.warning("Upload PDFs and click Submit first.")
        return

    response = st.session_state.conversation_chain({"question": user_question})
    st.session_state.chat_history = response.get("chat_history", [])

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(":man_in_tuxedo:", message.content)
        else:
            st.write(":robot_face:", message.content)


def main():
    st.set_page_config(page_title="Chat with Documents", page_icon=":books:")

    styl = """
    <style>
      .stTextInput { position: fixed; bottom: 3rem; width: 70%; }
    </style>
    """
    st.markdown(styl, unsafe_allow_html=True)

    st.title("AI Chat-Bot for Your Documents :books:")

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Upload files, click Submit, then ask questions:")
    if user_input:
        handle_userinput(user_input)

    with st.sidebar:
        pdf_docs = st.file_uploader(
            "Upload your documents here", type=["pdf"], accept_multiple_files=True
        )

        if st.button("Submit"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
                return

            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)

                embeddings = get_embeddings()
                vectorstore = FAISS.from_texts(chunks, embeddings)  # ✅ Fix: FAISS now defined

                st.session_state.conversation_chain = get_conversational_chain(vectorstore)
                st.success("Done! Ask your questions in the input box.")



if __name__ == "__main__":
    main()
