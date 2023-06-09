"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import OpenAI, VectorDBQA
from langchain.vectorstores import Pinecone
import os
import pinecone

OPENAI_API_KEY =os.environ['OPENAI_API_KEY']
PINECONE_API_KEY =os.environ['PINECONE_API_KEY']
PINE_ENV = os.environ['PINE_ENV']




def load_chain():
    """Logic for loading the chain you want to use should go here."""
    # llm = OpenAI(temperature=0)
    # chain = ConversationChain(llm=llm)
    
    model_name = 'text-embedding-ada-002'

    embed = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY
    )

    index_name = 'testcanadapolicy'
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINE_ENV
    )


    text_field = "text"

    # switch back to normal index for langchain
    index = pinecone.Index(index_name)

    vectorstore = Pinecone(
        index, embed.embed_query, text_field
    )

    llm = OpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
    )

    qa = VectorDBQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        vectorstore=vectorstore
    )
    return qa

chain = load_chain()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Ask for Canadian Policy - Innovative Solutions Canada", page_icon=":robot:")
st.header("Ask for RFP Canadian Policy- Innovative Solutions Canada")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "", key="query")
    return input_text


user_input = get_text()

if user_input:
    #st.write(f"input dictionary: {{'query': {user_input}}}")
    #output = chain.run(inputs={"query": user_input})
    output = chain.run(user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        
