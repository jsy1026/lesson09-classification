
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------
# 페이지 설정
# -----------------------------
st.set_page_config(
    page_title="탐색 | 뇌졸중 예측 실습실",
    page_icon="🔎",
    layout="wide",
)

DATA_URL = "https://raw.githubusercontent.com/jsy1026/data/refs/heads/main/stroke.csv"


# -----------------------------
# 데이터 불러오기
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🔎 데이터 탐색")
st.write(
    "나이, 평균 혈당, 고혈압, 심장병, 체질량지수, 흡연 상태와 "
    "뇌졸중 사이의 데이터 분포와 비율을 살펴봅니다."
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


# -----------------------------
# 숫자 열 변환
# -----------------------------
for col in [
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease",
    "stroke",
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# -----------------------------
# 1. 나이와 평균 혈당의 분포
# -----------------------------
st.header("1. 나이와 평균 혈당의 분포")

hist_col1, hist_col2 = st.columns(2)

with hist_col1:
    age_data = df.dropna(subset=["age"])

    fig_age = px.histogram(
        age_data,
        x="age",
        nbins=30,
        title="나이의 분포",
        labels={
            "age": "나이",
            "count": "사람 수",
        },
    )

    fig_age.update_layout(
        xaxis_title="나이",
        yaxis_title="사람 수",
    )

    st.plotly_chart(
        fig_age,
        use_container_width=True,
    )


with hist_col2:
    glucose_data = df.dropna(
        subset=["avg_glucose_level"]
    )

    fig_glucose = px.histogram(
        glucose_data,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당의 분포",
        labels={
            "avg_glucose_level": "평균 혈당",
            "count": "사람 수",
        },
    )

    fig_glucose.update_layout(
        xaxis_title="평균 혈당",
        yaxis_title="사람 수",
    )

    st.plotly_chart(
        fig_glucose,
        use_container_width=True,
    )


# -----------------------------
# 2. 뇌졸중 여부별 나이와 평균 혈당 비교
# -----------------------------
st.header("2. 뇌졸중 여부에 따른 나이와 평균 혈당 비교")

stroke_df = df[
    df["stroke"].isin([0, 1])
].copy()

stroke_df["뇌졸중 여부"] = stroke_df["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음",
})

box_col1, box_col2 = st.columns(2)

# 나이 상자그림
with box_col1:
    age_box_data = stroke_df.dropna(
        subset=["age"]
    )

    fig_age_box = px.box(
        age_box_data,
        x="뇌졸중 여부",
        y="age",
        color="뇌졸중 여부",
        title="뇌졸중 여부별 나이",
        labels={
            "age": "나이",
            "뇌졸중 여부": "그룹",
        },
        category_orders={
            "뇌졸중 여부": [
                "뇌졸중 없음",
                "뇌졸중 있음",
            ]
        },
    )

    fig_age_box.update_layout(
        xaxis_title="뇌졸중 여부",
        yaxis_title="나이",
        showlegend=False,
    )

    st.plotly_chart(
        fig_age_box,
        use_container_width=True,
    )


# 평균 혈당 상자그림
with box_col2:
    glucose_box_data = stroke_df.dropna(
        subset=["avg_glucose_level"]
    )

    fig_glucose_box = px.box(
        glucose_box_data,
        x="뇌졸중 여부",
        y="avg_glucose_level",
        color="뇌졸중 여부",
        title="뇌졸중 여부별 평균 혈당",
        labels={
            "avg_glucose_level": "평균 혈당",
            "뇌졸중 여부": "그룹",
        },
        category_orders={
            "뇌졸중 여부": [
                "뇌졸중 없음",
                "뇌졸중 있음",
            ]
        },
    )

    fig_glucose_box.update_layout(
        xaxis_title="뇌졸중 여부",
        yaxis_title="평균 혈당",
        showlegend=False,
    )

    st.plotly_chart(
        fig_glucose_box,
        use_container_width=True,
    )


# 그룹별 평균값 표
mean_table = (
    stroke_df.groupby("뇌졸중 여부")
    .agg(
        사람 수=("stroke", "size"),
        평균 나이=("age", "mean"),
        평균 혈당=("avg_glucose_level", "mean"),
    )
    .reindex([
        "뇌졸중 없음",
        "뇌졸중 있음",
    ])
    .reset_index()
)

mean_table["평균 나이"] = mean_table[
    "평균 나이"
].round(2)

mean_table["평균 혈당"] = mean_table[
    "평균 혈당"
].round(2)

st.subheader("그룹별 평균값")

st.dataframe(
    mean_table,
    use_container_width=True,
    hide_index=True,
)


# -----------------------------
# 3. 고혈압과 심장병 여부별 뇌졸중 비율
# -----------------------------
st.header("3. 고혈압과 심장병 여부별 뇌졸중 비율")


def make_rate_table(data, feature, feature_name):
    valid = data[
        data[feature].isin([0, 1])
        & data["stroke"].isin([0, 1])
    ].copy()

    valid["구분"] = valid[feature].map({
        0: f"{feature_name} 없음",
        1: f"{feature_name} 있음",
    })

  result = (
    valid.groupby("구분")
    .agg({
        "stroke": ["count", "sum", "mean"]
    })
    .reset_index()
)

result.columns = [
    "구분",
    "사람 수",
    "뇌졸중 환자 수",
    "뇌졸중 비율",
]

    result["뇌졸중 비율(%)"] = (
        result["뇌졸중 비율"] * 100
    )

    return result


rate_col1, rate_col2 = st.columns(2)


# 고혈압 여부별 비율
with rate_col1:
    hypertension_table = make_rate_table(
        df,
        "hypertension",
        "고혈압",
    )

    fig_hypertension = px.bar(
        hypertension_table,
        x="구분",
        y="뇌졸중 비율(%)",
        text=hypertension_table[
            "뇌졸중 비율(%)"
        ].map(lambda x: f"{x:.2f}%"),
        title="고혈압 여부별 뇌졸중 비율",
        labels={
            "구분": "고혈압 여부",
            "뇌졸중 비율(%)": "뇌졸중 비율(%)",
        },
    )

    fig_hypertension.update_traces(
        textposition="outside"
    )

    fig_hypertension.update_layout(
        yaxis_rangemode="tozero",
    )

    st.plotly_chart(
        fig_hypertension,
        use_container_width=True,
    )

    st.dataframe(
        hypertension_table[
            [
                "구분",
                "사람 수",
                "뇌졸중 환자 수",
                "뇌졸중 비율(%)",
            ]
        ].round(2),
        use_container_width=True,
        hide_index=True,
    )


# 심장병 여부별 비율
with rate_col2:
    heart_table = make_rate_table(
        df,
        "heart_disease",
        "심장병",
    )

    fig_heart = px.bar(
        heart_table,
        x="구분",
        y="뇌졸중 비율(%)",
        text=heart_table[
            "뇌졸중 비율(%)"
        ].map(lambda x: f"{x:.2f}%"),
        title="심장병 여부별 뇌졸중 비율",
        labels={
            "구분": "심장병 여부",
            "뇌졸중 비율(%)": "뇌졸중 비율(%)",
        },
    )

    fig_heart.update_traces(
        textposition="outside"
    )

    fig_heart.update_layout(
        yaxis_rangemode="tozero",
    )

    st.plotly_chart(
        fig_heart,
        use_container_width=True,
    )

    st.dataframe(
        heart_table[
            [
                "구분",
                "사람 수",
                "뇌졸중 환자 수",
                "뇌졸중 비율(%)",
            ]
        ].round(2),
        use_container_width=True,
        hide_index=True,
    )


# -----------------------------
# 4. BMI 빈 값이 있는 사람들의 뇌졸중 비율
# -----------------------------
st.header("4. BMI 빈 값이 있는 사람들의 뇌졸중 비율")

valid_stroke = df[
    df["stroke"].isin([0, 1])
].copy()

missing_bmi = valid_stroke[
    valid_stroke["bmi"].isna()
]

# 전체 사람 통계
overall_count = len(valid_stroke)

overall_stroke_count = int(
    (valid_stroke["stroke"] == 1).sum()
)

# BMI 빈 값이 있는 사람 통계
missing_count = len(missing_bmi)

missing_stroke_count = int(
    (missing_bmi["stroke"] == 1).sum()
)

# 뇌졸중 비율 계산
overall_rate = (
    overall_stroke_count / overall_count * 100
    if overall_count > 0
    else 0
)

missing_rate = (
    missing_stroke_count / missing_count * 100
    if missing_count > 0
    else 0
)

bmi_compare = pd.DataFrame([
    {
        "대상": "BMI 빈 값이 있는 사람",
        "사람 수": missing_count,
        "뇌졸중 환자 수": missing_stroke_count,
        "뇌졸중 비율(%)": round(missing_rate, 2),
    },
    {
        "대상": "전체 사람",
        "사람 수": overall_count,
        "뇌졸중 환자 수": overall_stroke_count,
        "뇌졸중 비율(%)": round(overall_rate, 2),
    },
])

st.dataframe(
    bmi_compare,
    use_container_width=True,
    hide_index=True,
)


# -----------------------------
# 5. 흡연 상태별 사람 수
# -----------------------------
st.header("5. 흡연 상태별 사람 수")

smoking_counts = (
    df["smoking_status"]
    .fillna("빈 값")
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)

st.dataframe(
    smoking_counts,
    use_container_width=True,
    hide_index=True,
)


# -----------------------------
# 안내
# -----------------------------
st.caption(
    "참고: 뇌졸중 비율은 각 그룹에서 stroke가 1인 사람의 비율입니다. "
    "이 결과만으로 원인과 결과의 관계를 판단할 수는 없습니다."
)
