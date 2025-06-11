import streamlit as st
import pandas as pd
from data import Dataset
from engine import process_user_query

# Page config
st.set_page_config(page_title="Data Analyst Agent", layout="centered")
st.title("🤖 Data Analyst Agent")


# Cache dataset loading
@st.cache_data(show_spinner="Loading dataset, please wait...")
def load_bitext_dataset():
    ds = Dataset()
    return ds


if "response" not in st.session_state:
    st.session_state.response = ""

if "submitted" not in st.session_state:
    st.session_state.submitted = False

if "logs" not in st.session_state:
    st.session_state.logs = []

# Developer mode checkbox
st.sidebar.checkbox("Developer Mode", value=False, key="developer_mode")

if st.session_state.developer_mode:
    for message in st.session_state.logs:
        st.sidebar.text(message)


def log(message, print_to_sidebar=True):
    st.session_state.logs.append(message)
    print(message)

    if st.session_state.developer_mode and print_to_sidebar:
        st.sidebar.text(message)


# Initialize session state for data and messages
if "data" not in st.session_state:
    st.session_state.data = load_bitext_dataset()
    log("Dataset loaded and cached.")


def answer_my_question():
    if st.session_state.user_query.strip():
        st.session_state.submitted = True  # trigger UI update to show spinner
    else:
        st.warning("Please enter a question before submitting.")


with st.form("user_form", clear_on_submit=False, border=False):
    st.text_area(
        "Ask a question about the dataset:",
        key="user_query",
        height=100,
        disabled=st.session_state.submitted,
    )
    st.form_submit_button(
        "Answer my question",
        disabled=st.session_state.submitted,
        on_click=answer_my_question,
    )


def on_reset_click():
    st.session_state.data = load_bitext_dataset()
    st.session_state.user_query = ""
    st.session_state.response = ""
    st.session_state.submitted = False

    log("App state was reset (follwing user request).", print_to_sidebar=False)


if st.session_state.submitted:
    if not st.session_state.response:
        with st.spinner("Processing your question..."):
            log(f"User asked: {st.session_state.user_query}")
            output = process_user_query(
                st.session_state.user_query,
                st.session_state.data,
                log,
            )

            st.session_state.data = output["dataset"]
            st.session_state.response = output["response"]
            log(f"Generated response: '{st.session_state.response}'")

    st.markdown("### 💬 Agent Response")
    st.write(st.session_state.response)

    st.button("Ask a new question", on_click=on_reset_click)


# Optional developer info dump
if st.session_state.developer_mode:
    with st.expander("Session State", expanded=False):
        st.json(dict(st.session_state))


# st.set_page_config(page_title="Data Analyst Agent", layout="centered")
# st.title("🤖 Data Analyst Agent")

# # Session state init
# if "response" not in st.session_state:
#     st.session_state.response = ""

# if "answered" not in st.session_state:
#     st.session_state.answered = False

# if "developer_mode" not in st.session_state:
#     st.session_state.developer_mode = True

# if "reset_triggered" not in st.session_state:
#     st.session_state.reset_triggered = False

# # Developer mode toggle
# developer_mode = st.sidebar.checkbox("Developer Mode", key="developer_mode")


# # Logging helper
# def log(message):
#     if st.session_state.developer_mode:
#         st.sidebar.text(message)
#     else:
#         print(message)


# # Load and cache dataset
# @st.cache_data(show_spinner=True)
# def load_bitext_dataset():
#     return Dataset()


# if "data" not in st.session_state:
#     st.session_state.data = load_bitext_dataset()
#     log("Dataset loaded and cached.")

# if st.session_state.reset_triggered:
#     st.session_state.user_query = ""
#     st.session_state.reset_triggered = False

# # === FORM-BASED INTERACTION ===
# with st.form("user_form", clear_on_submit=False):
#     user_query = st.text_area(
#         "Ask a question about the dataset:",
#         key="user_query",
#         height=100,
#         disabled=st.session_state.answered,
#     )
#     submitted = st.form_submit_button(
#         "Answer my question", disabled=st.session_state.answered
#     )

# # Process input after form is submitted
# if submitted:
#     if user_query.strip():
#         with st.spinner("Processing your question..."):
#             log(f"User asked: {user_query}")
#             output = process_user_query(user_query, st.session_state.data, log)

#             st.session_state.data = output["dataset"]
#             st.session_state.response = output["response"]
#             st.session_state.answered = True
#             log(f"Generated response: {st.session_state.response}")
#     else:
#         st.warning("Please enter a question before submitting.")

# # Reset button (after form)
# if st.button("Reset", disabled=not st.session_state.answered):
#     st.session_state.reset_triggered = True
#     st.session_state.response = ""
#     st.session_state.answered = False
#     st.session_state.data = load_bitext_dataset()
#     log("App state reset.")

# # Show output
# if st.session_state.response:
#     st.markdown("### 💬 Agent Response")
#     st.write(st.session_state.response)

# # Developer info
# if developer_mode:
#     with st.expander("Session State", expanded=False):
#         st.json(dict(st.session_state))
