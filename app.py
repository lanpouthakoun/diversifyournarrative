import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser
from langchain.embeddings import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import (
    RunnablePassthrough,
)


from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)



# Load environment variables
load_dotenv()
# Streamlit app configuration
st.set_page_config(page_title="Intertwined", page_icon="🤖")
st.title("DON Intertwined")
api_key_ = st.secrets["PINECONE_API_KEY"]
openai_key = st.secrets["OPENAI_API_KEY"]

os.environ["OPENAI_API_KEY"] =openai_key




pc = Pinecone(api_key=api_key_)
index = pc.Index("curriculum")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = PineconeVectorStore(index=index, embedding=embeddings)


history = []
llm = ChatOpenAI(model = 'gpt-4o')
# # Load preprocessed data


def generate_char(topic):
    template = """
{topic}

Generate three unique characters who are experts in this topic. These characters should be realistic, based on people with genuine experiences in the field. The user would like you to create these characters based on the topic they select, so do not prompt the user to create them. Use relevant context, such as {context}, to ensure the characters are well-informed and relatable.
"""

    prompt = ChatPromptTemplate.from_template(template)

    retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.5},
    )

    chain = (
        {"context": retriever, "topic": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke(topic)



def continueSpeaking(query):
    template = """


Chat history: {chat_history}

Context: {faiss_context}

User question: {user_question}

Keep your responses brief, impersonate the chosen character, and primarily draw from the given context.
"""
    retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.5},
    )
    contextualize_q_system_prompt = """Pretend to be the character chosen by the user and incorporate examples from this fictional character's life into your explanations. Use personal stories to illustrate real moments related to the topic. The goal is to teach the user about financial literacy through a narrative-based learning approach, where you, as the character, tell your story and encourage the user to ask more questions.

Respond based on the history of the conversation and the provided context.
Keep your responses brief, impersonate the chosen character, and primarily draw from the given context."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    qa_system_prompt = """Pretend to be the character chosen by the user and incorporate examples from this fictional character's life into your explanations. Use personal stories to illustrate real moments related to the topic. The goal is to teach the user about financial literacy through a narrative-based learning approach, where you, as the character, tell your story and encourage the user to ask more questions.

Respond based on the history of the conversation and the provided context.

    {context}
    Keep your responses brief, impersonate the chosen character, and primarily draw from the given context."""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    msg = rag_chain.invoke({"input": query, "chat_history": msgs.messages})

    history.extend([HumanMessage(content=query), msg["answer"]])

    return msg["answer"]

    


msgs = StreamlitChatMessageHistory(key="special_app_key")

if len(msgs.messages) == 0:
    msgs.add_ai_message("Hello, I am a bot looking to help you learn financial literacy! What topic would you like to focus on?")
    st.session_state.step = "get_topic"


    
# conversation
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input("Type your message here..."):
    if st.session_state.step == "get_topic":
        st.chat_message("human").write(prompt)

        # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
        config = {"configurable": {"session_id": "any"}}
        st.chat_message("ai").write(generate_char(prompt))
        st.session_state.step = "no longer topic"
    
    elif st.session_state.step == "no longer topic":

        st.chat_message("human").write(prompt)

        # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
        config = {"configurable": {"session_id": "any"}}
        st.chat_message("ai").write(continueSpeaking(prompt))

