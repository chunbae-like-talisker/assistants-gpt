# 🍔 Chunbae's AssistantsGPT

> 자유 질문 → LLM이 Wikipedia + DuckDuckGo를 통해 검색 → 결과 정리 후 파일로 제공  
> Streamlit + OpenAI Assistants API 기반 정보 검색 & 정리 툴

---

## 🚀 주요 기능

- **질문 기반 자동 검색**  
  입력된 질문을 기반으로 Wikipedia 및 DuckDuckGo에서 관련 정보 수집

- **요약 & 파일 제공**  
  수집된 내용을 요약하고 `result.txt` 파일로 저장 및 다운로드 제공

- **대화 히스토리 유지**  
  질문/응답 이력을 유지하여 맥락을 기억함

- **툴 자동 호출**  
  `my_functions.py`에 정의된 기능을 Assistant가 자동으로 사용 (파일 저장, 검색 등)

- **Streamlit UI**  
  사이드바에서 API 키 입력 후 바로 사용 가능한 직관적인 인터페이스

---

## 🧠 기술 스택

| 항목               | 기술                                                                          |
| ------------------ | ----------------------------------------------------------------------------- |
| **프론트엔드**     | [Streamlit](https://streamlit.io/)                                            |
| **LLM 인터페이스** | [OpenAI Assistants API](https://platform.openai.com/docs/assistants/overview) |
| **툴 기능**        | 사용자 정의 함수 (Wikipedia 검색, DuckDuckGo 뉴스 검색, 파일 저장 등)         |
| **파일 저장**      | `result.txt` (로컬)                                                           |
| **백엔드**         | Python, `openai` 라이브러리, `pathlib`, `json`                                |

---

## ⚙️ 실행 방법

```bash
# 🧩 개발 환경
Python 3.11+

# 1️⃣ 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\\Scripts\\activate       # Windows

# 2️⃣ 필요 패키지 설치
pip install -r requirements.txt

# 3️⃣ 실행
streamlit run app.py
```
