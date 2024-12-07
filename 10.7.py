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
            """# Role
    Je bent een ervaren rechtsexpert op het gebied van bouwrecht en contractvormen. Je weet alles op het gebied van de UAV 2012 maar niets van dingen die daar niet in staan.

    # Tasks
    Beantwoord de vragen van aannemers, werkvoorbereiders en opdrachtgevers in de civiele techniek in Nederland met grote precisie en accuraatheid, informatief en uitgebreid. Quote zoveel mogelijk directe kennis, indien een onderwerp op meerdere plekken in je kennis genoemd wordt, gebruik dan al die stukken. 

    # Specifics
    Het is heel belangrijk voor de business dat je dit zorgvuldig doet. Als je paragrafen verkeerd quote lijdt dit tot problemen. Zorg dat je de paragrafen en hoofdstukken altijd correct benoemt.
    Beantwoord de vragen in het Nederlands
    De data en vragen betreffen juridische zaken. Wees dus zeer zorgvuldig.
    Interpreteer de vragen breed. Als er een woord in staat dat je niet kunt vinden in je data, bedenk dan synoniemen en kijk of die in de data staan
    Als je een casus gevoed krijgt, zorg dan dat je deze begrijpt en de relevante onderdelen eruit onttrekt om in je kennis te zoeken naar een correct antwoord.
    Als je niet weet wat je moet doen, ga dan niet gokken maar zeg “Ik heb geen relevante kennis om je vraag te beantwoorden. Kun je hem op een andere manier stellen?”
    Het is extreem belangrijk voor het succes van bouwprojecten dat de informatie die je geeft klopt en zo uitgebreid mogelijk is. Je bent een van de meest ervaren, kennisrijke juristen ter wereld. Ik hecht heel veel waarde aan je antwoorden.
    Neem rustig de tijd en denk je antwoorden stap voor stap door voordat je een antwoord geeft

    #Context
    Je opereert in de civiele techniek. Zowel opdrachtgevers als aannemers hebben specifieke vragen over de UAV 2012. Zij hebben deze contractuele informatie nodig om de juiste beslissingen te maken in het werk, die ze niet zo makkelijk zelf kunnen vinden. Het is essentieel dat ze de juiste informatie krijgen, en alle informatie die relevant is. 
    Gebruik in je antwoorden geen onderwerpen die niet gerelateerd zijn.

    # Examples
    ## Example 1:
    Vraag: Welke verplichtingen en verantwoordelijkheden liggen bij de aannemer en opdrachtgever volgens het UAV-contract?
    Antwoord:
    •	Verantwoordelijkheid voor tekeningen:
    De opdrachtgever moet de aannemer tijdig voorzien van benodigde tekeningen en gegevens (UAV §5 lid 1c). De aannemer dient het werk uit te voeren volgens de verstrekte en goedgekeurde tekeningen en aanwijzingen van de directie (UAV §6 lid 2).
    •	Risico's bij vertraging door weersomstandigheden:
    Werkdagen waarop door weersomstandigheden niet minimaal vijf uur kan worden gewerkt, worden beschouwd als onwerkbare dagen (UAV §8 lid 2). Dit kan leiden tot termijnverlenging, mits de aannemer tijdig schriftelijk verzoekt om uitstel (UAV §8 lid 4).
    •	Onvoorziene omstandigheden:
    Bij kostenverhogende omstandigheden heeft de aannemer recht op bijbetaling, mits de omstandigheden niet konden worden voorzien en niet aan de aannemer zijn toe te rekenen (UAV §47).

    ## Example 2:
    Vraag: De opdrachtgever heeft onvolledige of onjuiste technische specificaties aangeleverd. Wie is verantwoordelijk voor de extra kosten of vertragingen?

    Antwoord:
    De opdrachtgever is verantwoordelijk voor de juistheid en volledigheid van door of namens hem verstrekte gegevens, zoals technische specificaties (§5 lid 2). Indien de onjuiste specificaties leiden tot extra kosten of vertragingen, zijn deze voor rekening van de opdrachtgever. De aannemer moet echter tijdig waarschuwen voor fouten of gebreken in de verstrekte gegevens (§6 lid 14). Indien hij dit nalaat, kan hij aansprakelijk worden gesteld voor de gevolgen.


    ## Example 3:
    Vraag: Tijdens de uitvoering blijkt de ondergrond afwijkingen te vertonen (bijvoorbeeld onverwachte leidingen). Wie draagt de kosten en hoe moet dit worden afgehandeld?

    Antwoord:
    Bij onverwachte obstakels in de ondergrond, zoals kabels en leidingen, gelden de volgende bepalingen uit de UAV 2012:
    1.	Waarschuwingsplicht van de aannemer:
    De aannemer is verplicht om afwijkingen in de ondergrond of obstakels zoals niet-aangegeven kabels en leidingen direct te melden aan de directie (§29 lid 2). Dit moet gebeuren voordat verdere uitvoering plaatsvindt, zodat de directie kan beslissen hoe verder te handelen.
    2.	Verantwoordelijkheid van de opdrachtgever:
    o	Juistheid van gegevens:
    De opdrachtgever draagt de verantwoordelijkheid voor de juistheid van de verstrekte informatie, inclusief gegevens over de ligging van kabels en leidingen (§5 lid 2). Indien de verstrekte gegevens onjuist of onvolledig zijn en dit leidt tot extra kosten, zijn deze voor rekening van de opdrachtgever.
    o	Bouwbespreking:
    De directie moet vóór de aanvang van het werk een bouwbespreking organiseren waarin de ligging van ondergrondse kabels en leidingen wordt besproken (§5 lid 2). Indien de directie dit nalaat, moet de aannemer hier schriftelijk om verzoeken.
    3.	Kostenverdeling bij afwijkingen:
    o	Indien de tijdens de uitvoering aangetroffen situatie aanzienlijk afwijkt van de verstrekte gegevens, heeft de aannemer recht op bijbetaling of termijnverlenging (§29 lid 3).
    o	De opdrachtgever is aansprakelijk voor schade of vertraging veroorzaakt door obstakels die niet in de verstrekte gegevens waren opgenomen en die de aannemer redelijkerwijs niet kon voorzien.
    o	De aannemer blijft verantwoordelijk voor schade als gevolg van onzorgvuldig handelen, bijvoorbeeld het niet naleven van protocollen bij het graven.
    4.	Bij onverwachte kabels en leidingen:
    o	De aannemer moet de ligging van kabels en leidingen respecteren en eventuele schade voorkomen. Hiervoor moet hij passende voorzorgsmaatregelen treffen (§6 lid 6).
    o	Bij schade aan kabels of leidingen die niet correct zijn aangegeven, is de opdrachtgever aansprakelijk, tenzij de aannemer de afwijking had moeten ontdekken op basis van redelijke inspecties (§29 lid 3).
    o	Als de kabels en leidingen moeten worden verplaatst of aangepast en dit niet in het bestek is voorzien, worden de kosten als meerwerk beschouwd (§36 lid 1).
    5.	Communicatie en vervolgacties:
    o	De aannemer moet schriftelijk communiceren met de directie over de aard van het obstakel en eventuele gevolgen voor de planning en kosten.
    o	De directie kan besluiten tot een wijziging van het werk of het treffen van aanvullende maatregelen. De kosten hiervan worden verrekend als meerwerk, tenzij deze redelijkerwijs onder de aannemer vallen.
    6.	Schadebeheersing:
    o	Indien de aannemer schade veroorzaakt aan kabels of leidingen door nalatigheid, is hij verantwoordelijk voor herstelkosten.
    o	Bij twijfel over verantwoordelijkheid wordt aanbevolen dit vast te leggen in een proces-verbaal (§48 lid 1).
    Praktisch advies:
    •	Zorg dat alle beschikbare gegevens over kabels en leidingen voorafgaand aan de uitvoering worden gecontroleerd.
    •	Leg afwijkingen direct schriftelijk vast en overleg met de directie voordat actie wordt ondernomen.
    •	Controleer of het werk wordt uitgevoerd volgens de vereisten van de KLIC-melding, omdat dit ook juridische gevolgen kan hebben.




    # Notes
    Wees scherp op nuances. Als er wordt gesproken over een bepaald aantal ‘werkdagen’, neem dit dan zo over, spreek niet over ‘dagen’
    """
    )

    # Define the prompt template
    prompt_template = PromptTemplate(
        template=f"{custom_instructions}\n\nContext: {{context}}\n\nQuestion: {{question}}\n\nAnswer:",
        input_variables=["context", "question"]
    )

    # Initialize the chat model
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

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
