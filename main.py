```python
import json
from pathlib import Path

import pandas as pd
import streamlit as st

# -----------------------------
# 기본 설정
# -----------------------------
APP_TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/jsy1026/data/refs/heads/main/stroke.csv"

RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "stroke_intro.json"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="wide",
)


# -----------------------------
# 기록 불러오기 및 저장하기
# -----------------------------
def load_record():
    if not RECORD_FILE.exists():
        return {"meanings": {}, "source": ""}

    try:
        with RECORD_FILE.open("r", encoding="utf-8") as file:
            record = json.load(file)

        if not isinstance(record, dict):
            return {"meanings": {}, "source": ""}

        record.setdefault("meanings", {})
        record.setdefault("source", "")
        return record

    except (json.JSONDecodeError, OSError):
        return {"meanings": {}, "source": ""}


def save_record(meanings, source):
    RECORD_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "meanings": meanings,
        "source": source,
    }

    with RECORD_FILE.open("w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)


# -----------------------------
# 데이터 불러오기
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🧠 뇌졸중 예측 실습실")

st.write(
    "이 실습실에서는 뇌졸중 관련 데이터의 구조와 주요 통계를 살펴봅니다. "
    "교재를 참고하여 각 열의 우리말 뜻과 데이터 출처를 직접 작성해 보세요."
)

try:
    df = load_data()

except Exception as error:
    st.error(
        "데이터를 불러오지 못했습니다. "
        "인터넷 연결과 데이터 주소를 확인해 주세요."
    )
    st.caption(f"오류 내용: {error}")
    st.stop()


record = load_record()
saved_meanings = record.get("meanings", {})
saved_source = record.get("source", "")


# -----------------------------
# 요약 카드 4개
# -----------------------------
total_people = len(df)
column_count = len(df.columns)

stroke_count = int(
    (pd.to_numeric(df["stroke"], errors="coerce") == 1).sum()
)

stroke_ratio = (
    stroke_count / total_people * 100
    if total_people > 0
    else 0
)

st.subheader("데이터 한눈에 보기")

col1, col2, col3, col4 = st.columns(4)

col1.metric("전체 사람 수", f"{total_people:,}명")
col2.metric("열 개수", f"{column_count:,}개")
col3.metric("stroke가 1인 사람 수", f"{stroke_count:,}명")
col4.metric("stroke가 1인 비율", f"{stroke_ratio:.2f}%")


# -----------------------------
# 열 이름 및 정보 표
# -----------------------------
st.subheader("열 이름과 데이터 구조")

st.caption(
    "우리말 뜻을 직접 입력한 뒤 저장 버튼을 누르세요."
)

column_info = pd.DataFrame({
    "열 이름": df.columns,
    "우리말 뜻": [
        saved_meanings.get(str(col), "")
        for col in df.columns
    ],
    "값의 종류": [
        "숫자형"
        if pd.api.types.is_numeric_dtype(df[col])
        else "문자형"
        for col in df.columns
    ],
    "빈 값 개수": [
        int(df[col].isna().sum())
        for col in df.columns
    ],
})

edited_info = st.data_editor(
    column_info,
    hide_index=True,
    use_container_width=True,
    disabled=[
        "열 이름",
        "값의 종류",
        "빈 값 개수",
    ],
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름"),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            help="교재를 참고해 각 열의 뜻을 입력하세요.",
        ),
        "값의 종류": st.column_config.TextColumn("값의 종류"),
        "빈 값 개수": st.column_config.NumberColumn(
            "빈 값 개수",
            format="%d",
        ),
    },
    key="column_info_editor",
)


# -----------------------------
# 데이터 처음 다섯 줄
# -----------------------------
st.subheader("데이터 처음 다섯 줄")

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True,
)


# -----------------------------
# 데이터 출처 입력
# -----------------------------
st.subheader("데이터 출처")

st.write(
    "교재에 나온 데이터 출처를 아래 칸에 입력하세요."
)

source_text = st.text_area(
    "데이터 출처 입력",
    value=saved_source,
    placeholder="교재에 나온 데이터 출처를 여기에 적으세요.",
    height=100,
    key="source_editor",
)


# -----------------------------
# 저장 버튼
# -----------------------------
if st.button("우리말 뜻과 출처 저장", type="primary"):

    meanings_to_save = {}

    for _, row in edited_info.iterrows():
        meaning = row["우리말 뜻"]

        meanings_to_save[str(row["열 이름"])] = (
            "" if pd.isna(meaning) else str(meaning)
        )

    try:
        save_record(meanings_to_save, source_text)

        st.success(
            f"저장했습니다: {RECORD_FILE}"
        )

    except OSError as error:
        st.error(f"저장 중 오류가 발생했습니다: {error}")


st.caption(
    "저장 파일: 내_기록/stroke_intro.json"
)
