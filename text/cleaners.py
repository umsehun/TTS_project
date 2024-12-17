import re

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

def collapse_whitespace(text):
    """여러 공백을 하나로 축소"""
    return re.sub(_whitespace_re, ' ', text).strip()

def korean_cleaners(text):
    """
    한국어 전처리 함수.
    - 특수문자 제거
    - 공백 정리
    """
    text = re.sub(r"[^가-힣0-9\s.,!?]", "", text)  # 한글, 숫자, 기본 특수문자만 허용
    text = collapse_whitespace(text)  # 공백 정리
    return text

def basic_cleaners(text):
    """소문자 변환 및 공백 축소"""
    text = text.lower()  # 한국어에는 필요 없으나 영어 섞인 데이터가 있을 경우 사용
    text = collapse_whitespace(text)
    return text
