import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

# URL 텍스트 추출
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=5) # url에 HTTP 요청을 보낸다. -> 해당 url의 html 데이터를 받아온다.
        soup = BeautifulSoup(response.text, "html.parser") # HTML 데이터를 BeautifulSoup 객체로 만든다.
        for tag in soup(["script", "style"]): tag.decompose() # 필요없는 태그 데이터 제거
        return soup.get_text(separator="\n").strip() # 텍스트 데이터만 추출하여 반환
    except Exception:
        return ""

# PDF 텍스트 추출
def extract_text_from_pdf(file):
    text = ""
    try:
        reader = PdfReader(file) # pdf 파일 로드
        for page in reader.pages: # 각 페이지에 대해서 텍스트 추출
            text += page.extract_text() or "" # 문서 내용을 text에 추가
        return text
    except Exception:
        return ""