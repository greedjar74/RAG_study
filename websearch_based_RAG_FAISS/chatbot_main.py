import streamlit as st
from PIL import Image

from rag_chatbot import rag_chatbot

def main_page():
    st.title('스무디')
    st.markdown('## 기업 및 개인 맞춤 면접 에이전트')
    st.markdown('### ✨스무디는 당신의 취업 성공을 기원합니다!✨')
    
    image = Image.open('websearch_based_RAG_FAISS/image.png')
    st.image(image, caption="취업 성공을 기원합니다!")

page_names_to_funcs = {"Main Page": main_page, 'chat-bot': rag_chatbot}
selected_page = st.sidebar.selectbox('Select a page', page_names_to_funcs)

page_names_to_funcs[selected_page]()