from src.logger import logger
from src.helper import load_env, create_documents, create_embeddings
import streamlit as st
from src.utils import generate_session_id
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.prompt import contextulized_q_system_prompt, system_prompt

### Load environment variables
load_env()
st.set_page_config(
    page_title="Chatbot",
    page_icon="🤖",
    layout="centered",
)
st.title("Chatbot")
st.write("This is a conversational chatbot application.")

api_key=st.text_input("Enter GROQ key:", type="password")

if api_key:
    st.session_state.api_key = api_key
    logger.info("API key entered successfully.")
    
    model=st.selectbox(
        "Select a model",
        options=["gemma2-9b-it", "llama-3.1-8b-instant"],
        index=0,
    )
    if model:
        llm=ChatGroq(model_name=model, api_key=st.session_state.api_key)
    else:
        st.warning("Please select a model to continue.")
    
    # Initialize session state variables
    if "store" not in st.session_state:
        st.session_state.store = {}
    
    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"], accept_multiple_files=True)
    
    #create docs
    docs=create_documents(uploaded_file)
    logger.info(f"Documents created: {len(docs)}")
    
    # Create embeddings
    vectorstore = create_embeddings(docs)
    logger.info("Embeddings created successfully.")
    
    retriver=vectorstore.as_retriever()
    logger.info("Retriever created successfully.")
        
    contextualized_q_promt = ChatPromptTemplate.from_messages(
        [
            ("system", contextulized_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriver = create_history_aware_retriever(llm,retriver, contextualized_q_promt)
    logger.info("History aware retriever created successfully.")
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    logger.info("QA chain created successfully.")
    
    rag_chain = create_retrieval_chain(history_aware_retriver, qa_chain)
    logger.info("Retrieval chain created successfully.")
     
    session_id=st.text_input("Enter a session ID:", value=generate_session_id(), key="session_id")
    if not session_id:
        st.warning("Please enter a session ID to continue.")
        
   
    store = {}


    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]
    
     
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    user_input = st.text_input("Ask a question:")
    if user_input:
        session_history = get_session_history(session_id)
        with st.spinner("Generating response..."):
            try:
                result = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={
                        "configurable": {"session_id": session_id}
                    }
                
                )
                 
                st.write(st.session_state.store)
                st.write(result["answer"])
                st.write("Chat History:", session_history.messages)
                
            except Exception as e:
                logger.error(f"Error occurred: {e}")
                st.error("An error occurred while generating the response.")
    
    
else:
    st.warning("Please enter your API key to continue.")
