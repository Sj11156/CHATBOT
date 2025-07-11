import streamlit as st
import os
from io import BytesIO
import base64
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain

# --- Configuration ---
# CHANGE 1: Use the multimodal model LLaVA
MODEL_NAME = "llava"
MAX_HISTORY = 5
os.environ['OLLAMA_HOST'] = '0.0.0.0:11434'

# --- LangChain Components ---
llm = ChatOllama(model=MODEL_NAME)
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=MAX_HISTORY
)

# 3. Create a Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant. You can analyze images and reply based on provided context."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ]
)

# 4. Create the Chain
conversation_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

# --- Streamlit UI ---

st.set_page_config(page_title="Ollama Multimodal Chatbot", layout="centered")
st.title("Local LLM Chatbot (Vision Capable)")

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # CHANGE 2: Display image if present in the message
        if "content" in message:
            st.markdown(message["content"])
        if "image" in message:
            st.image(message["image"], width=200)

# --------------------------
# Image Upload Section (CHANGE 3)
# --------------------------
uploaded_file = st.sidebar.file_uploader("Upload an image (optional)", type=['png', 'jpg', 'jpeg'])

# React to user input
if prompt_input := st.chat_input("Ask me anything or about the image..."):

    # Store the user's text input
    user_message_content = prompt_input

    # Define the input structure for LangChain's LLMChain.invoke()
    langchain_input = {"question": user_message_content}

    # --------------------------
    # 4. Image Handling and Base64 Encoding
    # --------------------------
    if uploaded_file is not None:
        # Read the image file content
        image_bytes = uploaded_file.read()

        # Convert image bytes to base64 format (required by Ollama for image input)
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        # Format the input for the multimodal model:
        # LangChain's Ollama integration expects a list of dictionaries with type 'text' or 'image_url'

        # This structure goes into the "question" variable of the LLMChain input
        langchain_input["question"] = [
            {"type": "text", "text": user_message_content},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]

        # Add the image path to the Streamlit session state for display in the chat history
        st.session_state.messages.append({"role": "user", "content": user_message_content, "image": image_bytes})
    else:
        # If no image uploaded, just add the text content to the Streamlit session state
        st.session_state.messages.append({"role": "user", "content": user_message_content})

    # Display the user message in the chat
    with st.chat_message("user"):
        st.markdown(user_message_content)
        if uploaded_file is not None:
            st.image(uploaded_file, width=200)

    # --------------------------
    # 5. Invoke the LLM
    # --------------------------

    # Get response from the LLM chain
    with st.spinner("Thinking..."):
        try:
            # Invoke the chain with the current input (text and/or image)
            # This works for both text-only and multimodal input
            response = conversation_chain.invoke(langchain_input)
            ai_response = response["text"]
        except Exception as e:
            ai_response = f"An error occurred: {e}"
            st.error(ai_response)

    # --------------------------
    # 6. Handle LLM Response
    # --------------------------

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(ai_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})