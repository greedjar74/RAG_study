import streamlit as st

from nowcoding_RAG import nowcoding_RAG
from db_check import db_check

def main_page():
    st.title('나우코딩랩스 RAG 실습')
    st.text('RAG를 활용한 실습을 진행하세요.')

page_names_to_funcs = {"Main Page": main_page, 'RAG 실습': nowcoding_RAG, 'DB 확인': db_check}
selected_page = st.sidebar.selectbox('Select a page', page_names_to_funcs)

page_names_to_funcs[selected_page]()