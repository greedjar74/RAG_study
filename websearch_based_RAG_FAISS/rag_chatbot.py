import streamlit as st
import os
import sys

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from funcs.generate_interview_questions import generate_interview_questions

st.set_page_config(layout="centered")

# 텍스트 분할기
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Streamlit UI
def rag_chatbot():
    st.title("스무디")

    for key in ["questions", "current_q", "user_answers", "docs_used", "sources"]:
        if key not in st.session_state:
            st.session_state[key] = []

    # API 키 입력
    st.sidebar.header("🔐 API Key 설정")
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
    serper_key = st.sidebar.text_input("Serper.dev API Key", type="password")

    # 기본 정보 입력
    company_name = st.text_input("1️⃣ 지원할 기업명을 입력하세요", placeholder="예: 삼성전자, 카카오")
    pdf_file = st.file_uploader("2️⃣ 자기소개서 PDF를 업로드하세요", type=["pdf"])

    if not openai_key or not serper_key or not company_name or not pdf_file:
        st.warning("모든 항목을 입력하고 파일을 업로드해주세요.")
        st.stop()

    os.environ["OPENAI_API_KEY"] = openai_key
    sys.stdout.reconfigure(encoding="utf-8")

    # 질문 생성 버튼
    if st.button("🚀 면접 예상 질문 생성 시작"):
        with st.spinner("질문 생성 중..."):
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            response_text, docs_used, sources = generate_interview_questions(company_name, pdf_file, embeddings, serper_key)

            questions = [q.strip("-•● ").strip() for q in response_text.strip().split("\n") if q.strip()]
            st.session_state.questions = questions
            st.session_state.current_q = 0
            st.session_state.user_answers = []
            st.session_state.docs_used = docs_used
            st.session_state.sources = sources

            st.success("면접 예상 질문 생성 완료! 아래에서 시작하세요.")
            st.rerun()
    
    # 전체 질문 미리보기
    if st.session_state.questions:
        with st.expander("📋 생성된 전체 질문 목록 보기"):
            for idx, question in enumerate(st.session_state.questions, 1):
                st.markdown(f"**{idx}. {question}**")

    # 질문/응답 인터페이스
    if st.session_state.questions:
        curr_idx = st.session_state.current_q
        if curr_idx < len(st.session_state.questions):
            curr_q = st.session_state.questions[curr_idx]
            st.subheader(f"📝 질문 {curr_idx + 1}")
            st.markdown(f"**{curr_q}**")

            answer = st.text_area("✍️ 당신의 답변을 입력하세요", key=f"answer_{curr_idx}")

            if st.button("➡️ 다음 질문으로"):
                if answer.strip():
                    st.session_state.user_answers.append(answer.strip())
                    st.session_state.current_q += 1
                    st.rerun()
                else:
                    st.warning("답변을 입력해주세요.")

            # ✅ 이전 질문 및 답변 출력
            if st.session_state.user_answers:
                st.markdown("---")
                st.markdown("### 📌 이전 질문 및 답변")
                for i, (q, a) in enumerate(zip(
                    st.session_state.questions[:curr_idx],
                    st.session_state.user_answers
                ), 1):
                    st.markdown(f"**Q{i}: {q}**")
                    st.markdown(f"🗣 **답변:** {a}")
                    st.markdown("---")

        else:
            st.success("🎉 모든 질문에 답변하셨습니다!")

            for i, (q, a) in enumerate(zip(
                st.session_state.questions,
                st.session_state.user_answers
            ), 1):
                st.markdown(f"---\n**Q{i}: {q}**")
                st.markdown(f"🗣 **답변:** {a}")

            with st.expander("📄 참고 문서 보기"):
                for i, (doc, src) in enumerate(zip(st.session_state.docs_used, st.session_state.sources), 1):
                    preview = doc.page_content[:300].replace("\n", " ")
                    st.markdown(f"**[{i}]** [{src}]({src})\n\n{preview}...")