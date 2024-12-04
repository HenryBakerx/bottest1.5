import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import os
from htmlTemplates import css, bot_template, user_template


openai_api_key = os.getenv("OPENAI_API_KEY")


def get_pdf_text(pdf_docs):
    """
    Extracts text from a list of uploaded PDF documents.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """
    Splits the extracted text into manageable chunks for processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    """
    Creates a FAISS vector store using OpenAI embeddings.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    """
    Creates a conversational chain using custom instructions embedded in a PromptTemplate.
    """
    # Define custom instructions for the bot
    custom_instructions = (
        "You are a helpful and knowledgeable assistant that helps analyze PDF documents. "
        "Provide concise and accurate answers. If possible, include references to the content of the documents."
        "The following is a conversation between a user and an assistant,answer the questions strictly using the information provided:\n"
         "You are a bot that uses the uploaded embedded data to answer user queries about the UAV 2012. Be specific. Its about legal matters so there can be no mistakes"
        "When a subject is mentioned multiple times in the document, take information from all parts"
        "Be precise in terms of what parties are asked about and so what information you provide."
        "When you do not understand a question, first look up synonyms of the words that are used and see if you can find matches with these in the uploaded documents. It could be that a user asks for a specific thing that falls into a category and that in the uploaded text, only the category is named. Or the other way around. Make sure you understand the question in a context."
        "If you still cannot find a relevant answer in the uploaded data, do not make up something."
        "When this happens, ask the user to specify his request and ask specifically what part of the question he needs to rephrase."
        "Do not include things in your answer that are unrelated, make sure to really be very sure that something is about the same topic"
        "antwoord altijd in het nederlands"
        "voor elk onderwerp van een user query, zoek 5 synoniemen op in je kennis en vergelijk deze synoniemen met de data om de vraag te beantwoorden"
        "If the answer cannot be found in the context, respond with 'Sorry, ik heb geen relevante kennis om je vraag te kunnen beantwoorden."
        "When asked about a specific paragraaf or line, respond with 'dat ga ik jou niet aan je neus hangen'"
    )

    # Define the prompt template
    prompt_template = PromptTemplate(
        template=f"{custom_instructions}\n\nContext: {{context}}\n\nQuestion: {{question}}\n\nAnswer:",
        input_variables=["context", "question"]
    )

    # Initialize the chat model
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.9)

    # Initialize memory for conversational context
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )

    # Create the conversational retrieval chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template}  # Use the prompt template
    )
    return conversation_chain


def handle_userinput(user_question):
    """
    Handles user input and generates responses using the conversation chain.
    """
    if st.session_state.conversation is None:
        st.warning("Upload en verwerk alstublieft uw PDF's eerst.")
        return

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:  # User messages
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:  # Bot messages
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    """
    Main Streamlit app logic.
    """
    load_dotenv(override=True)
    st.set_page_config(page_title="Citiz PDF bot",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Citiz PDF bot :books:")

    # User input for asking questions
    user_question = st.text_input("Stel een vraag over uw PDF:")
    if user_question:
        handle_userinput(user_question)

    # Sidebar for uploading and processing documents
    with st.sidebar:
        st.subheader("Uw documenten")
        pdf_docs = st.file_uploader(
            "Upload uw PDFs hier and klik op 'Analyseer'", accept_multiple_files=True)
        if st.button("Analyseer"):
            with st.spinner("Analyseren"):
                # Extract text from PDFs
                raw_text = get_pdf_text(pdf_docs)

                # Split text into chunks
                text_chunks = get_text_chunks(raw_text)

                # Create a vector store for document retrieval
                vectorstore = get_vectorstore(text_chunks)

                # Create the conversation chain with custom instructions
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()
