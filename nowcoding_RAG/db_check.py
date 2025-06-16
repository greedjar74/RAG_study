from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import streamlit as st


def db_check():
    # Chroma DB 경로
    db_path = "./nowcoding_DB"

    # Vectorstore 로드
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
    )

    # 저장된 모든 문서 보기
    all_docs = vectorstore.get()['documents']

    # 출력
    st.text(f"저장된 문서 수: {len(all_docs)}\n")

    for i, doc in enumerate(all_docs[0:10], 1):  # 처음 10개만 출력
        st.text(f"--- 문서 {i} ---")
        st.text(doc[:1500])  # 500자까지 출력
