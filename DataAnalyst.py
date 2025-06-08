import streamlit as st
import pandas as pd
from data import Dataset
from engine import process_user_query

# Page config
st.set_page_config(page_title="Data Analyst Agent", layout="centered")
st.title("🤖 Data Analyst Agent")


# Cache dataset loading
@st.cache_data(show_spinner=True)
def load_bitext_dataset():
    ds = Dataset()
    return ds


if "response" not in st.session_state:
    st.session_state.response = ""

if "answered" not in st.session_state:
    st.session_state.answered = False

# Developer mode checkbox
developer_mode = st.sidebar.checkbox("Developer Mode", value=True, key="developer_mode")


def log(message):
    if st.session_state.developer_mode:
        st.sidebar.text(message)
    else:
        print(message)


# Initialize session state for data and messages
if "data" not in st.session_state:
    st.session_state.data = load_bitext_dataset()
    log("Dataset loaded and cached.")


# User input
# user_query = st.text_input("Ask a question about the dataset:", key="user_query")
user_query = st.text_area(
    "Ask a question about the dataset:",
    key="user_query",
    height=100,
    disabled=st.session_state.answered,
)

# Buttons
col1, col2 = st.columns([1, 1])

with col1:

    def on_answer_click():
        if st.session_state.user_query.strip():
            log(f"User asked: {st.session_state.user_query}")
            output = process_user_query(
                st.session_state.user_query, st.session_state.data, log
            )

            st.session_state.data = output["dataset"]
            st.session_state.response = output["response"]
            log(f"Generated response: {st.session_state.response}")

            st.session_state.answered = True
        else:
            st.warning("Please enter a question before submitting.")

    st.button(
        "Answer my question",
        disabled=st.session_state.answered,
        on_click=on_answer_click,
    )

with col2:

    def on_reset_click():
        st.session_state.data = load_bitext_dataset()
        st.session_state.user_query = ""
        st.session_state.response = ""
        st.session_state.answered = False

        # Optionally clear developer_mode checkbox as well:
        # st.session_state.developer_mode = True

        log("App state reset.")

    st.button("Reset", disabled=not st.session_state.answered, on_click=on_reset_click)

# Output display
if st.session_state.response:
    st.markdown("### 💬 Agent Response")
    st.write(st.session_state.response)

# Optional developer info dump
if developer_mode:
    with st.expander("Session State", expanded=False):
        st.json(dict(st.session_state))
