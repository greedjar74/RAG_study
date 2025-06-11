'''
V7 기반 chatbot 
- streamlit 사용
'''

import streamlit as st
import os
import sys
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# 🌐 페이지 설정
st.set_page_config(page_title="교육과정 이론 챗봇", layout="wide")
st.title("📘 교육과정 이론 기반 챗봇")

# 🔑 OpenAI API 설정
st.sidebar.header("🔐 API 설정")
api_key_input = st.sidebar.text_input("OpenAI API Key", type="password")
if not api_key_input:
    st.warning("사이드바에 API 키를 입력해주세요.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key_input
sys.stdout.reconfigure(encoding="utf-8")

# PDF 문서 로딩
pdf_file_path = "4월 22일 커리큘럼 - 홍석.pdf"
reader = PdfReader(pdf_file_path)
text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

# 문서 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]

# Chroma DB 설정
db_path = "./hong_curriculum_DB"
if os.path.exists(db_path):
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
    )
else:
    vectorstore = Chroma.from_documents(
        docs,
        OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=db_path
    )
    vectorstore.persist()

# RAG LLM 및 Retriever
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model="gpt-4.1-mini")

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
당신은 전문적인 도움을 주는 어시스턴트입니다.
다음 문서 내용을 참고하여 사용자 질문에 구체적으로 답해주세요.

[문서 내용]
{context}

[대화 내용 및 질문]
{question}

[답변]
"""
)

# 문서 포맷 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 🔄 대화 저장 함수
def store_chat_to_chroma(user_input, assistant_reply):
    conversation_text = f"user: {user_input}\nassistant: {assistant_reply}"
    doc = Document(page_content=conversation_text)
    vectorstore.add_documents([doc])
    vectorstore.persist()

# RAG 실행 함수
def run_RAG(user_input, chat_history):
    docs = retriever.invoke(user_input)
    context = format_docs(docs)
    conversation = '\n'.join(chat_history + [f'user: {user_input}'])
    prompt_text = prompt.format(context=context, question=conversation)
    response = llm.invoke(prompt_text)
    return response.content, docs

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 이전 대화 출력 (Chroma에 저장되었지만 화면에는 출력하지 않음)
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["message"])

# 사용자 입력 받기
user_input = st.chat_input("질문을 입력하세요")

if user_input:
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        # 응답 생성
        response_text, source_docs = run_RAG(user_input, [f'{m["role"]}: {m["message"]}' for m in st.session_state.chat_history])

        # 응답 표시
        with st.chat_message("assistant"):
            st.markdown(response_text)

            # 참고 문서 표시
            with st.expander("📄 참고한 문서 내용 보기"):
                for i, doc in enumerate(source_docs, 1):
                    st.markdown(f"**[{i}]** {doc.page_content.strip()[:500]}...")

        # 대화 세션 및 DB 저장
        st.session_state.chat_history.append({"role": "user", "message": user_input})
        st.session_state.chat_history.append({"role": "assistant", "message": response_text})
        store_chat_to_chroma(user_input, response_text)

    except Exception as e:
        st.error(f"⚠️ 오류 발생: {e}")
