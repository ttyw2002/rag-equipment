import glob

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

# .env 파일에서 API 키를 읽음 (로컬 실행용)
# Streamlit Cloud에서는 Secrets에 등록한 값이 자동으로 적용됩니다
load_dotenv()

st.set_page_config(page_title="장비 오류 이력 검색", layout="wide")
st.title("장비 오류 이력 검색")
st.caption("단어가 정확히 일치하지 않아도 의미가 비슷한 이력을 찾아옵니다.")


# ---------- 1. 데이터 로드 + 벡터 스토어 구축 ----------
# @st.cache_resource : 앱이 새로고침돼도 딱 한 번만 실행되게 함
# 이게 없으면 검색할 때마다 임베딩을 새로 만들어 느리고 비쌈
@st.cache_resource
def build_vectorstore():
    # 폴더 안의 모든 엑셀을 자동으로 읽어 합침
    files = sorted(glob.glob("*.xlsx"))
    dfs = []
    for f in files:
        tmp = pd.read_excel(f, sheet_name="오류이력")
        tmp["출처파일"] = f
        dfs.append(tmp)
    df = pd.concat(dfs, ignore_index=True)

    documents = []
    for _, row in df.iterrows():
        # 검색 대상이 되는 본문 - 맥락을 붙여 서술형으로 구성
        content = (
            f"장비: {row['장비ID']} ({row['장비종류']}) | 공정: {row['공정']}\n"
            f"증상: {row['증상']}\n"
            f"원인: {row['원인분석']}\n"
            f"조치: {row['조치내역']}"
        )
        # 필터링·출처 표시에 쓸 구조화 정보
        meta = {
            "이력ID": row["이력ID"],
            "발생일시": str(row["발생일시"]),
            "장비ID": row["장비ID"],
            "장비종류": row["장비종류"],
            "공정": row["공정"],
            "에러코드": row["에러코드"],
            "다운타임_분": int(row["다운타임_분"]),
            "증상": row["증상"],
            "원인분석": row["원인분석"],
            "조치내역": row["조치내역"],
        }
        documents.append(Document(page_content=content, metadata=meta))

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return FAISS.from_documents(documents, embeddings), df


@st.cache_resource
def get_llm():
    # -latest 별칭은 구글이 최신 버전으로 자동 연결해줌
    return ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.1)


def to_text(raw):
    """최신 모델은 content를 리스트로 반환하기도 함 → 텍스트만 추출"""
    if isinstance(raw, list):
        return "\n".join(
            block.get("text", "")
            for block in raw
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return raw


with st.spinner("이력 데이터를 준비하는 중입니다. 처음 한 번만 시간이 걸립니다."):
    vectorstore, df = build_vectorstore()

st.success(f"이력 {len(df)}건 준비 완료")


# ---------- 2. 검색 화면 ----------
query = st.text_input(
    "무엇을 찾으시나요?",
    placeholder="예: 압력 관련 오류 이력",
)

col1, col2 = st.columns(2)
with col1:
    equip = st.selectbox("장비종류", ["전체"] + sorted(df["장비종류"].unique().tolist()))
with col2:
    topk = st.slider("결과 개수", 3, 15, 5)


# ---------- 3. 검색 + 답변 생성 ----------
if query:
    # Retrieval - 질문과 의미가 유사한 이력 검색
    if equip == "전체":
        results = vectorstore.similarity_search(query, k=topk)
    else:
        results = vectorstore.similarity_search(
            query, k=topk, filter={"장비종류": equip}
        )

    st.divider()
    tab1, tab2 = st.tabs(["AI 요약", f"이력 원본 {len(results)}건"])

    # Augmented + Generation - 검색된 이력을 근거로 LLM이 답변 생성
    with tab1:
        context = "\n\n".join(
            f"[{d.metadata['이력ID']}] {d.page_content}" for d in results
        )

        prompt = f"""당신은 반도체 공정 장비의 유지보수 이력을 분석하는 엔지니어입니다.
아래 이력만을 근거로 질문에 답하세요.

규칙:
1. 반드시 이력ID를 함께 인용하세요
2. 이력에 없는 내용은 추측하지 말고 "이력에 없음"이라고 하세요
3. 공통 원인이 보이면 묶어서 설명하세요
4. 5문장 이내로 간결하게 답하세요

--- 이력 ---
{context}

--- 질문 ---
{query}

--- 답변 ---"""

        with st.spinner("이력을 분석하는 중입니다."):
            try:
                answer = to_text(get_llm().invoke(prompt).content)
                st.write(answer)
            except Exception as e:
                st.error(f"답변 생성에 실패했습니다: {e}")

        st.caption("원본 탭에서 실제 이력을 확인하세요.")

    # 검색된 이력 원본
    with tab2:
        for doc in results:
            m = doc.metadata
            with st.container(border=True):
                st.markdown(f"**{m['이력ID']}** · {m['장비ID']} · `{m['에러코드']}`")
                st.write(f"**증상** — {m['증상']}")
                st.write(f"**원인** — {m['원인분석']}")
                st.write(f"**조치** — {m['조치내역']}")
                st.caption(f"{m['발생일시']} | {m['공정']} | 다운타임 {m['다운타임_분']}분")