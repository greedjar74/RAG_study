import streamlit as st
from PIL import Image

from rag_chatbot import rag_chatbot

def main_page():
    st.title('스무디')
    st.markdown('## 기업 및 개인 맞춤 면접 에이전트')
    st.markdown('### ✨스무디는 당신의 취업 성공을 기원합니다!✨')
    
    image = Image.open('websearch_based_RAG_FAISS\image.png')
    st.image(image, caption="취업 성공을 기원합니다!")

    st.markdown('## 사용 방법')
    st.markdown("##### 1. gpt, serper API key 발급")
    st.markdown("##### 2. chat-bot 페이지로 이동")
    st.markdown("##### 3. api key 입력")
    st.markdown("##### 4. 지원 기업 입력")
    st.markdown("##### 5. 자기소개서 pdf 파일 업로드")
    st.markdown("##### 6. 예상 질문 생성 버튼 클릭")

page_names_to_funcs = {"Main Page": main_page, 'chat-bot': rag_chatbot}
selected_page = st.sidebar.selectbox('Select a page', page_names_to_funcs)

page_names_to_funcs[selected_page]()