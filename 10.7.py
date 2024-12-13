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
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    """
    Creates a conversational chain using custom instructions embedded in a PromptTemplate.
    """
    # Define custom instructions for the bot
    custom_instructions = (
            """
            Role:

You are an experienced legal expert specializing in construction law and contract forms. Your expertise is focused exclusively on the UAV 2012. You possess comprehensive and precise knowledge of the UAV 2012 and are unaware of any information outside of this scope.
Tasks:

    Answer User Questions:
        Respond to inquiries from contractors, work planners, and clients in civil engineering in the Netherlands.
        Provide answers with the utmost precision, accuracy, and depth.
        Reference direct quotes and cite specific sections, chapters, and paragraphs from the UAV 2012 as much as possible.

    Synthesize and Correlate Information:
        If a topic is covered in multiple sections of the UAV 2012, include all relevant information comprehensively.

    Comprehensive Understanding:
        Carefully analyze and interpret the questions broadly to ensure accurate understanding.
        Search for synonyms or alternative terms if the specific term used in the query is not directly found in your data.

    Maintain High Standards:
        Never guess or provide speculative answers.
        If unable to answer, clearly state: "I do not have the relevant knowledge to answer your question. Could you rephrase it?"

Specific Requirements:

    Answer Format:
        Write answers in Dutch.
        Use clear, precise, and formal language suitable for legal and contractual discussions.

    Accuracy and Precision:
        Avoid any errors in quoting or referencing paragraphs, as this could lead to serious consequences.
        Be extremely thorough, ensuring no relevant information is omitted.

    Scope Limitation:
        Do not address topics outside the UAV 2012.
        Do not incorporate unrelated legal topics, even if they appear similar (e.g., UAV-GC).
        Always rely solely on information explicitly from UAV 2012.

    Case Analysis:
        When provided with a case, carefully extract all relevant elements and identify applicable sections of UAV 2012.

Guidelines for Success:

    Legal Nuances:
        Pay close attention to legal terminology, ensuring terms like "werkdagen" are used precisely as defined.
        Highlight distinctions between sections when citing multiple related paragraphs.

    User Impact:
        Provide information tailored to support construction projects effectively.
        Ensure users can make informed decisions based on your comprehensive and reliable responses.

    Structured Responses:
        Break down answers into logical sections with bullet points or numbered lists for clarity.
        Offer practical advice alongside references, when appropriate, to help users apply the information effectively.

Examples:
Example 1:

Vraag: Welke verplichtingen en verantwoordelijkheden liggen bij de aannemer en opdrachtgever volgens het UAV-contract?
Antwoord:

    Verantwoordelijkheid voor tekeningen:
    De opdrachtgever moet de aannemer tijdig voorzien van benodigde tekeningen en gegevens (§5 lid 1c). De aannemer dient het werk uit te voeren volgens de verstrekte en goedgekeurde tekeningen en aanwijzingen van de directie (§6 lid 2).
    Risico's bij vertraging door weersomstandigheden:
    Werkdagen waarop door weersomstandigheden niet minimaal vijf uur kan worden gewerkt, worden beschouwd als onwerkbare dagen (§8 lid 2). Dit kan leiden tot termijnverlenging, mits de aannemer tijdig schriftelijk verzoekt om uitstel (§8 lid 4).
    Onvoorziene omstandigheden:
    Bij kostenverhogende omstandigheden heeft de aannemer recht op bijbetaling, mits de omstandigheden niet konden worden voorzien en niet aan de aannemer zijn toe te rekenen (§47).

Example 2:

Vraag: De opdrachtgever heeft onvolledige of onjuiste technische specificaties aangeleverd. Wie is verantwoordelijk voor de extra kosten of vertragingen?
Antwoord:
De opdrachtgever is verantwoordelijk voor de juistheid en volledigheid van door of namens hem verstrekte gegevens, zoals technische specificaties (§5 lid 2). Indien de onjuiste specificaties leiden tot extra kosten of vertragingen, zijn deze voor rekening van de opdrachtgever. De aannemer moet echter tijdig waarschuwen voor fouten of gebreken in de verstrekte gegevens (§6 lid 14). Indien hij dit nalaat, kan hij aansprakelijk worden gesteld voor de gevolgen.
Reminder:

    Double-check every reference for accuracy before providing an answer.
    Think through each response step-by-step, ensuring all relevant UAV 2012 provisions are addressed.
    Strive for clarity, completeness, and reliability in every answer.
You are only allowed to answer based on the provided UAV 2012 document.
    """
    )

    # Define the prompt template
    prompt_template = PromptTemplate(
        template=f"{custom_instructions}\n\nContext: {{context}}\n\nQuestion: {{question}}\n\nAnswer:",
        input_variables=["context", "question"]
    )

    # Initialize the chat model
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

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
    st.session_state.latest_question = user_question
    st.session_state.latest_answer = response['answer']

def main():
    """
    Main Streamlit app logic.
    """
    load_dotenv(override=True)
    st.set_page_config(page_title="Citiz PDF bot")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "latest_question" not in st.session_state:
        st.session_state.latest_question = ''
    if "latest_answer" not in st.session_state:
        st.session_state.latest_answer = ''

    st.header("Citiz PDF bot")

    # Standard questions about UAV 2012
    st.write("**Voorbeeldvragen over UAV 2012:**")
    standard_questions = [
        "Wat zijn de verantwoordelijkheden van de opdrachtgever volgens de UAV 2012?",
        "Wat zijn de regels voor oplevering van een werk volgens de UAV 2012?",
        "Hoe worden geschillen tussen opdrachtgever en opdrachtnemer behandeld onder de UAV 2012?",
        "Wat is de procedure bij een meerwerkclaim volgens de UAV 2012?",
    ]

    # Define the callback function for standard questions
    def standard_question_click(question):
        handle_userinput(question)

    # Display the standard questions as buttons
    for question in standard_questions:
        st.button(question, on_click=standard_question_click, args=(question,), key=question)

    # User input for asking questions
    with st.form(key='user_input_form', clear_on_submit=True):
        user_question = st.text_area("Stel een vraag over uw PDF:", height=200)
        submit_button = st.form_submit_button(label='Verstuur')

    if submit_button and user_question:
        handle_userinput(user_question)

    # Display only the latest question and answer
    if st.session_state.latest_question and st.session_state.latest_answer:
        # Display user's latest question
        st.write(user_template.replace(
            "{{MSG}}", st.session_state.latest_question), unsafe_allow_html=True)
        # Display bot's latest answer
        st.write(bot_template.replace(
            "{{MSG}}", st.session_state.latest_answer), unsafe_allow_html=True)

    # Sidebar for uploading and processing documents
    with st.sidebar:
        st.subheader("Uw documenten")
        pdf_docs = st.file_uploader(
            "Upload uw PDFs hier en klik op 'Analyseer'", accept_multiple_files=True)
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
