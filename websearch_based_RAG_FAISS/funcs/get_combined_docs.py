from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from funcs.search_web import search_web
from funcs.extract_text import extract_text_from_pdf, extract_text_from_url

# 텍스트 분할기
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# 문서 결합
def get_combined_docs(company_name, pdf_file, serper_key, k=10):
    urls = search_web(f"{company_name} 면접 후기 질문 합격 팁", num_results=3, api_key=serper_key) # 찾은 url 정보
    docs = []
    for url in urls:
        text = extract_text_from_url(url) # 각 url에서 텍스트 정보 추출
        if text:
            chunks = text_splitter.split_text(text) # 텍스트 데이터를 chunk단위로 분할
            docs.extend([Document(page_content=chunk, metadata={"source": url}) for chunk in chunks]) # 각 chunk를 docs에 추가

    pdf_text = extract_text_from_pdf(pdf_file) # pdf 파일에서 텍스트 추출
    if pdf_text:
        chunks = text_splitter.split_text(pdf_text) # pdf 정보를 chunk 단위로 분할
        docs.extend([Document(page_content=chunk, metadata={"source": "자기소개서"}) for chunk in chunks]) # 각 chunk를 docs에 추가

    if docs:
        return docs
    
    return []