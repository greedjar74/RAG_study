import streamlit as st

from rag_chatbot import rag_chatbot

def main_page():
    st.title('면접 준비 보조 에이전트')
    st.text('면접 절차 & 기출 질문을 준비하세요.')
    st.text('취업 성공을 기원합니다!')

page_names_to_funcs = {"Main Page": main_page, 'chat-bot': rag_chatbot}
selected_page = st.sidebar.selectbox('Select a page', page_names_to_funcs)

page_names_to_funcs[selected_page]()