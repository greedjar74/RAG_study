'''
# RAG chatbot 구현
- gpt api 기반 (gpt-4.1-mini)
- chroma db 사용
- langchain 활용
- 원하는 경우 사용자 입력을 DB에 저장할 수 있다.
- DB 내용 확인 가능
- 상당히 높은 정확도를 보인다.
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

def rag_chatbot():
    st.title("RAG 실습")

    # 🔑 OpenAI API 설정
    st.sidebar.header("🔐 API 설정")
    api_key_input = st.sidebar.text_input("OpenAI API Key", type="password")
    if not api_key_input:
        st.warning("사이드바에 API 키를 입력해주세요.")
        st.stop()

    threshold = st.sidebar.slider(label='Threshold 설정', min_value=0.0, max_value=1.0, value=0.5, step=0.1)

    os.environ["OPENAI_API_KEY"] = api_key_input
    sys.stdout.reconfigure(encoding="utf-8")

    # PDF 문서 로딩
    pdf_file_path = "커리큘럼 제작 참고자료.pdf"
    reader = PdfReader(pdf_file_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    # 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]

    # Chroma DB 설정
    db_path = "./curriculum_theory_DB"
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
    너는 문서를 참고하여 답변을 생성한다. 최대한 정확한 답변을 생성하기 위해 문서를 꼼꼼히 확인한다.

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

    # 사용자 입력 prompt 저장 함수
    def store_user_input_to_chroma(user_input):
        conversation_text = f"user: {user_input}"
        doc = Document(page_content=conversation_text)
        vectorstore.add_documents([doc])
        vectorstore.persist()

    # AI 답변 저장 함수
    def store_ai_reply_to_chroma(assistant_reply):
        conversation_text = f"assistant: {assistant_reply}"
        doc = Document(page_content=conversation_text)
        vectorstore.add_documents([doc])
        vectorstore.persist()

    # RAG 실행 함수
    def run_RAG(user_input, chat_history):
        # docs = retriever.invoke(user_input) # 유사도 점수 없음
        docs_and_scores = retriever.vectorstore.similarity_search_with_score(user_input, k=5)
        doc_list = []
        score_list = []

        for doc, score in docs_and_scores:
            if abs(1-score) <= threshold: # 문서와 질문 간의 유사도 점수가 threshold 값보다 작은 경우 사용 -> 값이 작을수록 유사도가 높다는 의미
                doc_list.append(doc)
                score_list.append(abs(1-score))

        context = format_docs(doc_list)
        conversation = '\n'.join(chat_history + [f'user: {user_input}'])
        prompt_text = prompt.format(context=context, question=conversation)
        response = llm.invoke(prompt_text)
        return response.content, doc_list, score_list

    # 세션 상태 초기화
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 이전 대화 출력 (Chroma에 저장되었지만 화면에는 출력하지 않음)
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg['role'] == 'assistant':
                st.code(msg['content'], language='python')
            else :
                st.text(msg["content"])

    # 사용자 입력 받기
    user_input = st.chat_input("질문을 입력하세요")

    if user_input:
        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.text(user_input)

        try:
            # 응답 생성
            response_text, source_docs, docs_scores = run_RAG(user_input, [f'{m["role"]}: {m["content"]}' for m in st.session_state.chat_history])

            # 응답 표시
            with st.chat_message("assistant"):
                st.code(response_text, language='python')

                with st.expander("📄 참고한 문서 내용 보기"):
                    for i, doc in enumerate(source_docs, 1):
                        st.text(f"**[{i}]** {doc.page_content.strip()[:500]}... \n 유사도: {docs_scores[i-1]}") # 관련 문서 내용 및 유사도 출력

            # 대화 세션 저장 (UI용)
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})

            # ✅ 마지막 대화 세션도 저장 (임시 저장용)
            st.session_state.last_user_input = user_input
            st.session_state.last_response = response_text

        except Exception as e:
            st.error(f"⚠️ 오류 발생: {e}")

    # ✅ 대화 저장 버튼 (대화 생성 이후에만 사용 가능)
    if "last_user_input" in st.session_state and st.button("💾 마지막 사용자 입력을 Chroma DB에 저장하기"):
        try:
            store_user_input_to_chroma(
                st.session_state.last_user_input
            )
            st.success("✅ 마지막 사용자 입력을 Chroma DB에 저장되었습니다.")
        except Exception as e:
            st.error(f"⚠️ 저장 중 오류 발생: {e}")

    if "last_response" in st.session_state and st.button("💾 마지막 AI 응답을 Chroma DB에 저장하기"):
        try:
            store_ai_reply_to_chroma(
                st.session_state.last_response
            )
            st.success("✅ 마지막 AI 응답을 Chroma DB에 저장되었습니다.")
        except Exception as e:
            st.error(f"⚠️ 저장 중 오류 발생: {e}")

    # 🔁 대화 리셋 버튼 (system 메시지 제외)
    # 🔁 대화 리셋 버튼
    if st.button("⚠️ 대화 리셋"):
        st.session_state.chat_history = []
        if "last_user_input" in st.session_state:
            del st.session_state.last_user_input
        if "last_response" in st.session_state:
            del st.session_state.last_response
        st.rerun()