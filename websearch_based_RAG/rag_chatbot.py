import streamlit as st
import os
import sys
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 🔧 검색 + 크롤링
def search_web(query, num_results=3, api_key=None):
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": num_results}
    response = requests.post("https://google.serper.dev/search", json=payload, headers=headers)
    results = response.json().get("organic", [])
    return [item["link"] for item in results]

def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style"]): tag.decompose()
        return soup.get_text(separator="\n").strip()
    except Exception:
        return ""

# 🔧 문서 처리 및 벡터화 (Chroma 제거)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def get_search_docs(query, embedding_model, serper_key, k=5):
    urls = search_web(query, num_results=3, api_key=serper_key)
    docs = []
    for url in urls:
        text = extract_text_from_url(url)
        if text:
            chunks = text_splitter.split_text(text)
            docs.extend([Document(page_content=chunk, metadata={"source": url}) for chunk in chunks])

    if not docs:
        return None, []

    texts = [doc.page_content for doc in docs]
    embeddings = embedding_model.embed_documents(texts)

    def simple_retriever(query):
        query_vec = embedding_model.embed_query(query)
        sims = cosine_similarity([query_vec], embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:k]
        return [docs[i] for i in top_indices]

    return simple_retriever, docs

# 🔧 RAG 수행
def run_RAG(user_input, chat_history, serper_key):
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    retriever_func, all_docs = get_search_docs(user_input, embedding_model, serper_key)
    if not retriever_func:
        return "검색된 정보가 부족하여 답변할 수 없습니다.", [], []

    docs = retriever_func(user_input)
    context = "\n\n".join(doc.page_content for doc in docs)
    conversation = "\n".join(chat_history + [f"user: {user_input}"])

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""너는 문서를 참고하여 질문에 답변하는 AI 비서야.

[문서 내용]
{context}

[질문과 대화 내용]
{question}

[답변]
"""
    )
    prompt_text = prompt_template.format(context=context, question=conversation)

    llm = ChatOpenAI(model="gpt-4.1-mini")
    response = llm.invoke(prompt_text)

    return response.content, docs, [doc.metadata.get("source", "출처 없음") for doc in docs]

# ✅ Streamlit UI
def rag_chatbot():
    st.title("🔍 실시간 검색 기반 RAG Chatbot")

    # 사이드바
    st.sidebar.header("🔐 API 설정")
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
    serper_key = st.sidebar.text_input("Serper.dev API Key", type="password")

    if not openai_key or not serper_key:
        st.warning("API 키를 모두 입력해주세요.")
        st.stop()

    os.environ["OPENAI_API_KEY"] = openai_key
    sys.stdout.reconfigure(encoding="utf-8")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 이전 대화 출력
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 사용자 입력
    user_input = st.chat_input("질문을 입력하세요")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("답변 생성 중..."):
            try:
                response_text, docs_used, sources = run_RAG(
                    user_input,
                    [f'{m["role"]}: {m["content"]}' for m in st.session_state.chat_history],
                    serper_key
                )

                with st.chat_message("assistant"):
                    st.markdown(response_text)

                    with st.expander("📄 참고한 웹 문서"):
                        for i, (doc, src) in enumerate(zip(docs_used, sources), 1):
                            preview = doc.page_content[:300].replace("\n", " ")
                            st.markdown(f"**[{i}]** [{src}]({src})\n\n`{preview}...`")

                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})

            except Exception as e:
                st.error(f"❌ 오류 발생: {e}")

    if st.button("🔁 대화 리셋"):
        st.session_state.chat_history = []
        st.rerun()