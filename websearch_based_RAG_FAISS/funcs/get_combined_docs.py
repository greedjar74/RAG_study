from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import FAISS

from funcs.search_web import search_web
from funcs.extract_text import extract_text_from_pdf, extract_text_from_url

# 문서 결합
def get_combined_docs(company_name, pdf_file, embedding_model, serper_key, k=10):
    # 텍스트 분할기
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    urls = search_web(f"{company_name} 면접 후기 질문 합격 팁", num_results=5, api_key=serper_key)
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