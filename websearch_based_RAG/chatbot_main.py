import streamlit as st

from rag_chatbot import rag_chatbot

def main_page():
    st.title('교육과정 이론 Chat bot')
    st.text('chat-bot을 활용하여 교육과정 이론을 공부하세요.')

page_names_to_funcs = {"Main Page": main_page, 'chat-bot': rag_chatbot}
selected_page = st.sidebar.selectbox('Select a page', page_names_to_funcs)

page_names_to_funcs[selected_page]()