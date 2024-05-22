import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import pickle
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Load environment variables
load_dotenv()
# Streamlit app configuration
st.set_page_config(page_title="Intertwined", page_icon="🤖")
st.title("DON Intertwined")

# Set OpenAI API Key



# Load preprocessed data
def load_preprocessed_data(text_file, faiss_file):
    if os.path.exists(text_file) and os.path.exists(faiss_file):
        try:
            with open(text_file, 'rb') as f:
                texts = pickle.load(f)
            # Load the FAISS index
            docsearch = FAISS.load_local(faiss_file, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
            return texts, docsearch
        except (EOFError, pickle.UnpicklingError) as e:
            st.error(f"Error loading preprocessed data: {e}")
            return [], None
    else:
        st.error("Preprocessed data file does not exist.")
        return [], None

texts, docsearch = load_preprocessed_data('preprocessed_texts.pkl', 'faiss_index')

# Function to get context from FAISS
def get_context_from_faiss(query):
    search_results = docsearch.similarity_search(query, k= 5)
    if search_results:
        return search_results[0].page_content
    else:
        return "Sorry, I don't have information on that topic."




def generate_char(topic, faiss_context):
    template = """
    {topic}, this is the topic of financial literacy the user would like to more about. 
    Generate three unique characters who are experts in this subject. Make sure that they are possibly real people who have real experiences.
    Make sure to create characters for the user, DO NOT let the user create one themselves. 
    Here is some context to draw from: {faiss_context} .
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(model="gpt-4o")
        
    chain = prompt | llm | StrOutputParser()
    
    return chain.stream({
        "faiss_context": faiss_context,
        "topic": topic,
    })

def continueSpeaking(query, faiss_context):
    template = """
    Pretend to be the character chosen by the user and bring examples from this fictional character's life into your explanations. For example, use stories from your life to show real moments of this subject.
    The goal is to teach the user about financial literacy by telling the story of a character
    and prompting the user to ask further questions to do a narrative-based learning approach. Answer the everything considering the history of the conversation and the given context:

    Chat history: {chat_history}

    Context : {faiss_context}

    User question: {user_question}

    Be brief in your responses, and impersonate the character you are chosen to be. Use a bulk of your information from the given context.
    """

    prompt = ChatPromptTemplate.from_template(template)
    context = st.session_state.chat_history

    llm = ChatOpenAI(model="gpt-4o")
        
    chain = prompt | llm | StrOutputParser()
    
    return chain.stream({
        "chat_history": context,
        "faiss_context": faiss_context,
        "user_question": query,
    })

    


# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot looking to help you learn financial literacy! What topic would you like to focus on?"),
    ]
    st.session_state.step = "get_topic"


    
# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    if st.session_state.step == "get_topic":
        
        with st.chat_message("Human"):
            st.markdown(user_query)

        context = get_context_from_faiss(user_query)
        
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("AI"):
            response = st.write_stream(generate_char(user_query, context))
        st.session_state.chat_history.append(AIMessage(content=response))
        st.session_state.step = "talk"
    else:

        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)
        context = get_context_from_faiss(user_query)
        with st.chat_message("AI"):
            response = st.write_stream(continueSpeaking(user_query, context))

        st.session_state.chat_history.append(AIMessage(content=response))