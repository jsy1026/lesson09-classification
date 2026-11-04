import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="탐색",
    page_icon="🔎",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/jsy1026/data/refs/heads/main/stroke.csv"

# -----------------------------
# 데이터 불러오기
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df


try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()


# -----------------------------
# 데이터 전처리
# -----------------------------
# 분석에 사용할 숫자형 열을 숫자로 변환
numeric_columns = [
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease",
    "stroke"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# 분석에 필요한 데이터가 없는 행 제외
df = df.dropna(subset=["age", "avg_glucose_level", "stroke"]).copy()

# 뇌졸중 여부를 보기 좋은 한글 이름으로 변환
df["stroke_label"] = df["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음"
})

# 고혈압 여부
df["hypertension_label"] = df["hypertension"].map({
    0: "고혈압 없음",
    1: "고혈압 있음"
})

# 심장병 여부
df["heart_disease_label"] = df["heart_disease"].map({
    0: "심장병 없음",
    1: "심장병 있음"
})


# -----------------------------
# 페이지 제목
# -----------------------------
st.title("🔎 데이터 탐색")
st.write(
    "뇌졸중 데이터의 분포와 특성을 그래프로 살펴봅니다."
)

st.caption(f"전체 데이터: {len(df):,}명")


# -----------------------------
# 1. 나이와 평균 혈당 분포
# -----------------------------
st.header("1. 나이와 평균 혈당 분포")

col1, col2 = st.columns(2)

with col1:
    fig_age = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={
            "age": "나이",
            "count": "사람 수"
        }
    )

    fig_age.update_layout(
        xaxis_title="나이",
        yaxis_title="사람 수"
    )

    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    fig_glucose = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={
            "avg_glucose_level": "평균 혈당",
            "count": "사람 수"
        }
    )

    fig_glucose.update_layout(
        xaxis_title="평균 혈당",
        yaxis_title="사람 수"
    )

    st.plotly_chart(fig_glucose, use_container_width=True)


# -----------------------------
# 2. 뇌졸중 여부에 따른 나이와 평균 혈당 비교
# -----------------------------
st.header("2. 뇌졸중 여부에 따른 나이와 평균 혈당 비교")

col1, col2 = st.columns(2)

with col1:
    fig_age_box = px.box(
        df,
        x="stroke_label",
        y="age",
        color="stroke_label",
        title="뇌졸중 여부에 따른 나이 비교",
        labels={
            "stroke_label": "뇌졸중 여부",
            "age": "나이"
        },
        category_orders={
            "stroke_label": ["뇌졸중 없음", "뇌졸중 있음"]
        }
    )

    fig_age_box.update_layout(
        xaxis_title="뇌졸중 여부",
        yaxis_title="나이",
        showlegend=False
    )

    st.plotly_chart(fig_age_box, use_container_width=True)

with col2:
    fig_glucose_box = px.box(
        df,
        x="stroke_label",
        y="avg_glucose_level",
        color="stroke_label",
        title="뇌졸중 여부에 따른 평균 혈당 비교",
        labels={
            "stroke_label": "뇌졸중 여부",
            "avg_glucose_level": "평균 혈당"
        },
        category_orders={
            "stroke_label": ["뇌졸중 없음", "뇌졸중 있음"]
        }
    )

    fig_glucose_box.update_layout(
        xaxis_title="뇌졸중 여부",
        yaxis_title="평균 혈당",
        showlegend=False
    )

    st.plotly_chart(fig_glucose_box, use_container_width=True)


# 두 그룹의 평균값 표
mean_table = (
    df.groupby("stroke_label")[["age", "avg_glucose_level"]]
    .mean()
    .reindex(["뇌졸중 없음", "뇌졸중 있음"])
    .rename(columns={
        "age": "평균 나이",
        "avg_glucose_level": "평균 혈당"
    })
    .round(2)
)

st.subheader("뇌졸중 여부에 따른 평균값")
st.dataframe(mean_table, use_container_width=True)


# -----------------------------
# 3. 고혈압 및 심장병 여부에 따른 뇌졸중 비율
# -----------------------------
st.header("3. 고혈압 및 심장병 여부에 따른 뇌졸중 비율")

# 고혈압별 뇌졸중 비율
hypertension_rate = (
    df.groupby("hypertension_label")["stroke"]
    .mean()
    .mul(100)
    .reset_index(name="뇌졸중 비율")
)

hypertension_rate["뇌졸중 비율"] = hypertension_rate["뇌졸중 비율"].round(2)

# 심장병별 뇌졸중 비율
heart_disease_rate = (
    df.groupby("heart_disease_label")["stroke"]
    .mean()
    .mul(100)
    .reset_index(name="뇌졸중 비율")
)

heart_disease_rate["뇌졸중 비율"] = heart_disease_rate["뇌졸중 비율"].round(2)

col1, col2 = st.columns(2)

with col1:
    fig_hypertension = px.bar(
        hypertension_rate,
        x="hypertension_label",
        y="뇌졸중 비율",
        title="고혈압 여부에 따른 뇌졸중 비율",
        labels={
            "hypertension_label": "고혈압 여부",
            "뇌졸중 비율": "뇌졸중 비율(%)"
        },
        text="뇌졸중 비율"
    )

    fig_hypertension.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig_hypertension.update_layout(
        xaxis_title="고혈압 여부",
        yaxis_title="뇌졸중 비율(%)",
        yaxis_ticksuffix="%"
    )

    st.plotly_chart(fig_hypertension, use_container_width=True)

with col2:
    fig_heart_disease = px.bar(
        heart_disease_rate,
        x="heart_disease_label",
        y="뇌졸중 비율",
        title="심장병 여부에 따른 뇌졸중 비율",
        labels={
            "heart_disease_label": "심장병 여부",
            "뇌졸중 비율": "뇌졸중 비율(%)"
        },
        text="뇌졸중 비율"
    )

    fig_heart_disease.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig_heart_disease.update_layout(
        xaxis_title="심장병 여부",
        yaxis_title="뇌졸중 비율(%)",
        yaxis_ticksuffix="%"
    )

    st.plotly_chart(fig_heart_disease, use_container_width=True)


# -----------------------------
# 4. BMI 결측 여부에 따른 뇌졸중 비율
# -----------------------------
st.header("4. BMI가 비어 있는 사람들의 뇌졸중 비율")

missing_bmi_df = df[df["bmi"].isna()].copy()

missing_bmi_stroke_rate = (
    missing_bmi_df["stroke"].mean() * 100
    if len(missing_bmi_df) > 0
    else 0
)

overall_stroke_rate = df["stroke"].mean() * 100

bmi_rate_table = pd.DataFrame({
    "구분": [
        "BMI가 비어 있는 사람",
        "전체 사람"
    ],
    "사람 수": [
        len(missing_bmi_df),
        len(df)
    ],
    "뇌졸중 비율(%)": [
        round(missing_bmi_stroke_rate, 2),
        round(overall_stroke_rate, 2)
    ]
})

st.dataframe(
    bmi_rate_table,
    use_container_width=True,
    hide_index=True
)

if len(missing_bmi_df) == 0:
    st.info("BMI가 비어 있는 사람이 없습니다.")


# -----------------------------
# 5. 흡연 상태별 사람 수
# -----------------------------
st.header("5. 흡연 상태별 사람 수")

smoking_count = (
    df["smoking_status"]
    .fillna("결측")
    .value_counts()
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)

st.dataframe(
    smoking_count,
    use_container_width=True,
    hide_index=True
)
