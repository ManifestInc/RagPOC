import streamlit as st
import openai
from openai import OpenAI
import time
import os
from typing import Optional

# Page config
st.set_page_config(
    page_title="Company RAG Chat POC",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "assistant_id" not in st.session_state:
    st.session_state.assistant_id = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "vector_store_id" not in st.session_state:
    st.session_state.vector_store_id = None

# Sidebar for configuration
st.sidebar.title("🔧 Setup")

# OpenAI API Key input
api_key = st.sidebar.text_input(
    "Open API Key", 
    type="password",
    help="Get your API key from https://platform.openai.com/api-keys"
)

if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to continue.")
    st.stop()

# Initialize OpenAI client
try:
    client = OpenAI(api_key=api_key)
except Exception as e:
    st.error(f"Error initializing OpenAI client: {e}")
    st.stop()

def upload_files_to_vector_store(uploaded_files):
    """Upload files and create vector store"""
    try:
        with st.spinner("Creating vector store and uploading files..."):
            # Create vector store
            vector_store = client.beta.vector_stores.create(
                name="Company Documents POC"
            )
            
            # Prepare files for upload
            file_streams = []
            for uploaded_file in uploaded_files:
                file_streams.append((uploaded_file.name, uploaded_file.getvalue()))
            
            # Upload files to vector store
            file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store.id,
                files=[
                    openai.OpenAI().files.create(
                        file=(name, content), 
                        purpose="assistants"
                    ) for name, content in file_streams
                ]
            )
            
            st.session_state.vector_store_id = vector_store.id
            return vector_store.id
            
    except Exception as e:
        st.error(f"Error uploading files: {e}")
        return None

def create_assistant(vector_store_id):
    """Create OpenAI assistant with file search capability"""
    try:
        assistant = client.beta.assistants.create(
            name="Company Data Assistant",
            instructions="""You are a helpful assistant that answers questions about company documents. 
            Use the uploaded files to provide accurate, detailed answers. 
            If you can't find relevant information in the documents, say so clearly.
            Always cite which document you're referencing when possible.""",
            model="gpt-4-turbo-preview",
            tools=[{"type": "file_search"}],
            tool_resources={
                "file_search": {
                    "vector_store_ids": [vector_store_id]
                }
            }
        )
        
        st.session_state.assistant_id = assistant.id
        return assistant.id
        
    except Exception as e:
        st.error(f"Error creating assistant: {e}")
        return None

def create_thread():
    """Create a new conversation thread"""
    try:
        thread = client.beta.threads.create()
        st.session_state.thread_id = thread.id
        return thread.id
    except Exception as e:
        st.error(f"Error creating thread: {e}")
        return None

def get_assistant_response(message: str) -> Optional[str]:
    """Get response from assistant"""
    try:
        # Add message to thread
        client.beta.threads.messages.create(
            thread_id=st.session_state.thread_id,
            role="user",
            content=message
        )
        
        # Run the assistant
        with st.spinner("Thinking..."):
            run = client.beta.threads.runs.create(
                thread_id=st.session_state.thread_id,
                assistant_id=st.session_state.assistant_id
            )
            
            # Wait for completion
            while run.status in ['queued', 'in_progress']:
                time.sleep(1)
                run = client.beta.threads.runs.retrieve(
                    thread_id=st.session_state.thread_id,
                    run_id=run.id
                )
            
            if run.status == 'completed':
                # Get the assistant's response
                messages = client.beta.threads.messages.list(
                    thread_id=st.session_state.thread_id
                )
                
                return messages.data[0].content[0].text.value
            else:
                st.error(f"Run failed with status: {run.status}")
                return None
                
    except Exception as e:
        st.error(f"Error getting response: {e}")
        return None

# Main app
st.title("📚 Company RAG Chat POC")
st.markdown("Upload your company documents and start chatting with your data!")

# File upload section
st.sidebar.markdown("### 📁 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Choose files (PDF, CSV, TXT)",
    accept_multiple_files=True,
    type=['pdf', 'csv', 'txt', 'md', 'docx']
)

# Setup button
if uploaded_files and st.sidebar.button("🚀 Setup RAG System", type="primary"):
    # Upload files and create vector store
    vector_store_id = upload_files_to_vector_store(uploaded_files)
    
    if vector_store_id:
        # Create assistant
        assistant_id = create_assistant(vector_store_id)
        
        if assistant_id:
            # Create thread
            thread_id = create_thread()
            
            if thread_id:
                st.sidebar.success("✅ RAG System Ready!")
                st.sidebar.markdown(f"**Files uploaded:** {len(uploaded_files)}")
            else:
                st.sidebar.error("Failed to create conversation thread")
        else:
            st.sidebar.error("Failed to create assistant")
    else:
        st.sidebar.error("Failed to upload files")

# Chat interface
if st.session_state.assistant_id and st.session_state.thread_id:
    st.markdown("### 💬 Chat with your documents")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        response = get_assistant_response(prompt)
        
        if response:
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
        else:
            st.error("Failed to get response from assistant")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        # Create new thread
        create_thread()
        st.experimental_rerun()

else:
    st.info("👆 Upload your documents in the sidebar and click 'Setup RAG System' to get started!")
    
    # Sample questions
    st.markdown("### 🔍 Example Questions You Can Ask:")
    st.markdown("""
    - "What's our Q4 revenue according to the financial reports?"
    - "Summarize the key points from the strategy document"
    - "What are the main risks mentioned in our risk assessment?"
    - "Find all mentions of budget allocations"
    - "What does the CSV data show about our customer demographics?"
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Status")
st.sidebar.markdown(f"**Assistant:** {'✅ Ready' if st.session_state.assistant_id else '❌ Not setup'}")
st.sidebar.markdown(f"**Thread:** {'✅ Active' if st.session_state.thread_id else '❌ Not created'}")
st.sidebar.markdown(f"**Vector Store:** {'✅ Ready' if st.session_state.vector_store_id else '❌ Not setup'}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Note:** This POC uses OpenAI's API. Your documents will be processed by OpenAI.")