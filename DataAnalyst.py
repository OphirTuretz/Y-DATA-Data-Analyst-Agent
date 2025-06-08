import streamlit as st
import pandas as pd
from data import Dataset
from engine import process_user_query

# Page config
st.set_page_config(page_title="Data Analyst Agent", layout="centered")
st.title("🤖 Data Analyst Agent")

# Developer mode checkbox
developer_mode = st.sidebar.checkbox("Developer Mode")


def log(message):
    if developer_mode:
        st.sidebar.text(message)
    else:
        print(message)


# Cache dataset loading
@st.cache_data(show_spinner=True)
def load_bitext_dataset():
    ds = Dataset()
    return ds


# Initialize session state for data and messages
if "data" not in st.session_state:
    st.session_state.data = load_bitext_dataset()
    log("Dataset loaded and cached.")

if "response" not in st.session_state:
    st.session_state.response = ""

# User input
# user_query = st.text_input("Ask a question about the dataset:", key="user_query")
user_query = st.text_area(
    "Ask a question about the dataset:", key="user_query", height=100
)

# Buttons
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Answer my question"):
        if user_query.strip():
            log(f"User asked: {user_query}")

            # Placeholder response logic
            # (f"(This is where the agent would answer your question: '{user_query}')")
            response = process_user_query(user_query, st.session_state.data, log)

            st.session_state.response = response
            log(f"Generated response: {response}")
        else:
            st.warning("Please enter a question before submitting.")

with col2:
    if st.button("Reset"):
        st.session_state.user_query = ""
        st.session_state.response = ""
        log("App state reset.")

# Output display
if st.session_state.response:
    st.markdown("### 💬 Agent Response")
    st.write(st.session_state.response)

# Optional developer info dump
if developer_mode:
    with st.expander("Session State", expanded=False):
        st.json(dict(st.session_state))
