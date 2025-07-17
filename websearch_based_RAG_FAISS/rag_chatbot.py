import streamlit as st
import os
import sys
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from pypdf import PdfReader

st.set_page_config(layout="centered")

# 웹 검색
def search_web(query, num_results=3, api_key=None):
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": num_results}
    response = requests.post("https://google.serper.dev/search", json=payload, headers=headers)
    return [item["link"] for item in response.json().get("organic", [])]

# URL 텍스트 추출
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style"]): tag.decompose()
        return soup.get_text(separator="\n").strip()
    except Exception:
        return ""

# PDF 텍스트 추출
def extract_text_from_pdf(file):
    text = ""
    try:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception:
        return ""

# 텍스트 분할기
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# 문서 결합
def get_combined_docs(company_name, pdf_file, embedding_model, serper_key, k=10):
    urls = search_web(f"{company_name} 면접 후기 질문 합격 팁", num_results=3, api_key=serper_key)
    docs = []
    for url in urls:
        text = extract_text_from_url(url)
        if text:
            chunks = text_splitter.split_text(text)
            docs.extend([Document(page_content=chunk, metadata={"source": url}) for chunk in chunks])

    pdf_text = extract_text_from_pdf(pdf_file)
    if pdf_text:
        chunks = text_splitter.split_text(pdf_text)
        docs.extend([Document(page_content=chunk, metadata={"source": "자기소개서"}) for chunk in chunks])

    if docs:
        vectorstore = FAISS.from_documents(docs, embedding_model)
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        return retriever, docs
    return None, []

# 면접 질문 생성
def generate_interview_questions(company_name, pdf_file, embedding_model, serper_key):
    retriever, all_docs = get_combined_docs(company_name, pdf_file, embedding_model, serper_key)
    if not retriever:
        return "❌ 정보가 충분하지 않습니다.", [], []

    query = f"{company_name} 기업 면접을 준비 중이야. 면접 예상 질문을 만들어줘."
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
너는 인사담당자처럼 면접 질문을 만들어주는 AI야.

[문서 내용]
{context}

[요청]
{question}

[면접 예상 질문]
"""
    )

    prompt_text = prompt_template.format(context=context, question=query)
    llm = ChatOpenAI(model="gpt-4.1-mini")
    response = llm.invoke(prompt_text)
    return response.content, docs, [doc.metadata.get("source", "출처 없음") for doc in docs]

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