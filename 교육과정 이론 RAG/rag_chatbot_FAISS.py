'''
기존 chatbot에서 FAISS로 DB 구현
'''

import streamlit as st
import os
import sys
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate

def rag_chatbot():
    st.title("RAG 실습 (FAISS 기반)")

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

    # FAISS 설정
    db_path = "./faiss_curriculum_index"
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")

    if os.path.exists(db_path):
        vectorstore = FAISS.load_local(db_path, embedding, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(docs, embedding)
        vectorstore.save_local(db_path)

    print(4)

    # Retriever 및 LLM 설정
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model="gpt-4.1-mini")

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
    너는 문서를 참고하여 답변을 생성한다.

    [문서 내용]
    {context}

    [대화 내용 및 질문]
    {question}

    [답변]
    """
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def store_user_input_to_faiss(user_input):
        doc = Document(page_content=f"user: {user_input}")
        vectorstore.add_documents([doc])
        vectorstore.save_local(db_path)

    def store_ai_reply_to_faiss(assistant_reply):
        doc = Document(page_content=f"assistant: {assistant_reply}")
        vectorstore.add_documents([doc])
        vectorstore.save_local(db_path)

    def run_RAG(user_input, chat_history):
        docs_and_scores = vectorstore.similarity_search_with_score(user_input, k=5)
        doc_list = []
        score_list = []

        for doc, score in docs_and_scores:
            if abs(1 - score) <= threshold:
                doc_list.append(doc)
                score_list.append(abs(1 - score))

        context = format_docs(doc_list)
        conversation = '\n'.join(chat_history + [f'user: {user_input}'])
        prompt_text = prompt.format(context=context, question=conversation)
        response = llm.invoke(prompt_text)
        return response.content, doc_list, score_list

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg['role'] == 'assistant':
                st.code(msg['content'], language='python')
            else:
                st.text(msg["content"])

    user_input = st.chat_input("질문을 입력하세요")

    if user_input:
        with st.chat_message("user"):
            st.text(user_input)

        try:
            response_text, source_docs, docs_scores = run_RAG(
                user_input, [f'{m["role"]}: {m["content"]}' for m in st.session_state.chat_history]
            )

            with st.chat_message("assistant"):
                st.code(response_text, language='python')
                with st.expander("📄 참고한 문서 내용 보기"):
                    for i, doc in enumerate(source_docs, 1):
                        st.text(f"**[{i}]** {doc.page_content.strip()[:500]}... \n 유사도: {docs_scores[i-1]}")

            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
            st.session_state.last_user_input = user_input
            st.session_state.last_response = response_text

        except Exception as e:
            st.error(f"⚠️ 오류 발생: {e}")

    if "last_user_input" in st.session_state and st.button("💾 마지막 사용자 입력을 FAISS DB에 저장하기"):
        try:
            store_user_input_to_faiss(st.session_state.last_user_input)
            st.success("✅ 마지막 사용자 입력이 저장되었습니다.")
        except Exception as e:
            st.error(f"⚠️ 저장 중 오류 발생: {e}")

    if "last_response" in st.session_state and st.button("💾 마지막 AI 응답을 FAISS DB에 저장하기"):
        try:
            store_ai_reply_to_faiss(st.session_state.last_response)
            st.success("✅ 마지막 AI 응답이 저장되었습니다.")
        except Exception as e:
            st.error(f"⚠️ 저장 중 오류 발생: {e}")

    if st.button("⚠️ 대화 리셋"):
        st.session_state.chat_history = []
        if "last_user_input" in st.session_state:
            del st.session_state.last_user_input
        if "last_response" in st.session_state:
            del st.session_state.last_response
        st.rerun()
