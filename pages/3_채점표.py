import pandas as pd
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


# --------------------------------------------------
# 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="채점표 | 뇌졸중 예측 실습실",
    page_icon="📝",
    layout="wide",
)

st.title("📝 분류 모델 채점표")
st.write(
    "테스트 데이터로 두 분류 모델의 예측 결과를 채점합니다. "
    "뇌졸중(stroke=1)을 양성, 뇌졸중 아님(stroke=0)을 음성으로 둡니다."
)

DATA_URL = "https://raw.githubusercontent.com/jsy1026/data/refs/heads/main/stroke.csv"

FEATURES = [
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease",
]

FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압 여부",
    "heart_disease": "심장병 여부",
}


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    df = df[
        [
            "id",
            "age",
            "avg_glucose_level",
            "bmi",
            "hypertension",
            "heart_disease",
            "stroke",
        ]
    ].copy()

    # 숫자형으로 변환
    for col in FEATURES + ["stroke"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 번호 순으로 정렬
    df = df.sort_values("id").reset_index(drop=True)

    return df


try:
    df = load_data()

except Exception as error:
    st.error(
        "데이터를 불러오지 못했습니다. "
        "인터넷 연결과 데이터 주소를 확인해 주세요."
    )
    st.caption(f"오류 내용: {error}")
    st.stop()


# --------------------------------------------------
# 데이터 나누기
# 번호 순으로 10명씩 묶고, 각 묶음의 앞 3명을 테스트용으로 고정
# --------------------------------------------------
df["묶음 번호"] = df.index // 10
df["묶음 안 순서"] = df.index % 10

test_df = df[df["묶음 안 순서"] < 3].copy()
train_df = df[df["묶음 안 순서"] >= 3].copy()

# BMI 결측치는 학습용 데이터의 중앙값으로 채움
bmi_median = train_df["bmi"].median()

train_df["bmi"] = train_df["bmi"].fillna(bmi_median)
test_df["bmi"] = test_df["bmi"].fillna(bmi_median)

# 입력 속성 또는 정답이 비어 있는 행 제외
train_df = train_df.dropna(subset=FEATURES + ["stroke"])
test_df = test_df.dropna(subset=FEATURES + ["stroke"])

X_train = train_df[FEATURES]
y_train = train_df["stroke"].astype(int)

X_test = test_df[FEATURES]
y_test = test_df["stroke"].astype(int)


# --------------------------------------------------
# 데이터 요약
# --------------------------------------------------
test_count = len(test_df)
test_stroke_count = int((y_test == 1).sum())

st.write(
    f"**테스트용 사람 수: {test_count:,}명 | "
    f"그중 실제 뇌졸중인 사람: {test_stroke_count:,}명**"
)

st.caption(
    f"학습용 {len(train_df):,}명, 테스트용 {test_count:,}명입니다. "
    f"BMI 결측치는 학습용 중앙값 {bmi_median:.1f}으로 채웠습니다."
)


# --------------------------------------------------
# 두 모델 학습
# --------------------------------------------------
logistic_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42,
)

logistic_model.fit(X_train, y_train)


tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
)

tree_model.fit(X_train, y_train)


# --------------------------------------------------
# 기준 모델
# 학습용 데이터에서 많은 쪽으로만 예측
# --------------------------------------------------
majority_class = int(y_train.value_counts().idxmax())

baseline_train_pred = [majority_class] * len(y_train)
baseline_test_pred = [majority_class] * len(y_test)


# --------------------------------------------------
# 테스트 데이터 예측
# --------------------------------------------------
logistic_test_pred = logistic_model.predict(X_test)
tree_test_pred = tree_model.predict(X_test)


# --------------------------------------------------
# 혼동행렬 계산
# 행: 실제 / 열: 예측
# 순서: [[TN, FP], [FN, TP]]
# --------------------------------------------------
logistic_cm = confusion_matrix(
    y_test,
    logistic_test_pred,
    labels=[0, 1],
)

tree_cm = confusion_matrix(
    y_test,
    tree_test_pred,
    labels=[0, 1],
)

baseline_cm = confusion_matrix(
    y_test,
    baseline_test_pred,
    labels=[0, 1],
)


# --------------------------------------------------
# 지표 계산
# --------------------------------------------------
def calculate_metrics(cm):
    tn, fp, fn, tp = cm.ravel()

    total = tn + fp + fn + tp

    accuracy = (
        (tp + tn) / total
        if total > 0
        else None
    )

    # 재현율 = TP / 실제 뇌졸중인 사람 수
    actual_positive = tp + fn

    recall = (
        tp / actual_positive
        if actual_positive > 0
        else None
    )

    # 정밀도 = TP / 뇌졸중이라고 예측한 사람 수
    predicted_positive = tp + fp

    if predicted_positive == 0:
        precision = None
        f1 = None
    else:
        precision = tp / predicted_positive

        # F1 = 2 * 정밀도 * 재현율 / (정밀도 + 재현율)
        if recall is None or precision + recall == 0:
            f1 = 0.0
        else:
            f1 = (
                2 * precision * recall
                / (precision + recall)
            )

    return {
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1": f1,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


logistic_metrics = calculate_metrics(logistic_cm)
tree_metrics = calculate_metrics(tree_cm)
baseline_metrics = calculate_metrics(baseline_cm)


def format_metric(value):
    if value is None:
        return "계산할 수 없음"

    return f"{value * 100:.2f}%"


# --------------------------------------------------
# 혼동행렬 표시 함수
# --------------------------------------------------
def show_confusion_matrix(cm, model_name):
    tn, fp, fn, tp = cm.ravel()

    matrix_table = pd.DataFrame(
        [
            [tn, fp, tn + fp],
            [fn, tp, fn + tp],
            [tn + fn, fp + tp, tn + fp + fn + tp],
        ],
        index=[
            "실제: 뇌졸중 아님",
            "실제: 뇌졸중",
            "열 합계",
        ],
        columns=[
            "예측: 뇌졸중 아님",
            "예측: 뇌졸중",
            "행 합계",
        ],
    )

    # 각 칸에 TP·FP·FN·TN 약자 표시
    matrix_table.iloc[0, 0] = f"TN: {tn:,}"
    matrix_table.iloc[0, 1] = f"FP: {fp:,}"
    matrix_table.iloc[1, 0] = f"FN: {fn:,}"
    matrix_table.iloc[1, 1] = f"TP: {tp:,}"

    # 합계 표시
    matrix_table.iloc[0, 2] = f"{tn + fp:,}"
    matrix_table.iloc[1, 2] = f"{fn + tp:,}"
    matrix_table.iloc[2, 0] = f"{tn + fn:,}"
    matrix_table.iloc[2, 1] = f"{fp + tp:,}"
    matrix_table.iloc[2, 2] = f"{tn + fp + fn + tp:,}"

    st.markdown(f"**{model_name} 혼동행렬**")
    st.dataframe(
        matrix_table,
        use_container_width=True,
    )


# --------------------------------------------------
# 혼동행렬
# --------------------------------------------------
st.subheader("1. 테스트 데이터 혼동행렬")

cm_col1, cm_col2 = st.columns(2)

with cm_col1:
    show_confusion_matrix(
        logistic_cm,
        "로지스틱 회귀(확률로 답하는 모델)",
    )

with cm_col2:
    show_confusion_matrix(
        tree_cm,
        "의사결정트리(질문으로 답하는 모델)",
    )


# --------------------------------------------------
# 지표 비교표
# --------------------------------------------------
st.subheader("2. 모델 성능 비교")

metrics_table = pd.DataFrame(
    [
        {
            "모델": "로지스틱 회귀(확률로 답하는 모델)",
            "정확도": format_metric(logistic_metrics["정확도"]),
            "재현율": format_metric(logistic_metrics["재현율"]),
            "정밀도": format_metric(logistic_metrics["정밀도"]),
            "F1": format_metric(logistic_metrics["F1"]),
        },
        {
            "모델": "의사결정트리(질문으로 답하는 모델)",
            "정확도": format_metric(tree_metrics["정확도"]),
            "재현율": format_metric(tree_metrics["재현율"]),
            "정밀도": format_metric(tree_metrics["정밀도"]),
            "F1": format_metric(tree_metrics["F1"]),
        },
        {
            "모델": "기준 모델(훈련용에서 많은 쪽으로만 답하는 모델)",
            "정확도": format_metric(baseline_metrics["정확도"]),
            "재현율": format_metric(baseline_metrics["재현율"]),
            "정밀도": format_metric(baseline_metrics["정밀도"]),
            "F1": format_metric(baseline_metrics["F1"]),
        },
    ]
)

st.dataframe(
    metrics_table,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "정확도 = (TP + TN) / 전체 사람 수 · "
    "재현율 = TP / 실제 뇌졸중인 사람 수 · "
    "정밀도 = TP / 뇌졸중이라고 예측한 사람 수 · "
    "F1 = 2 × 정밀도 × 재현율 / (정밀도 + 재현율)"
)


# --------------------------------------------------
# 놓친 환자 / 헛짚은 사람 목록
# --------------------------------------------------
st.subheader("3. 모델이 놓친 환자와 헛짚은 사람")

model_choice = st.selectbox(
    "목록을 확인할 모델을 선택하세요.",
    options=[
        "로지스틱 회귀(확률로 답하는 모델)",
        "의사결정트리(질문으로 답하는 모델)",
    ],
)

if model_choice == "로지스틱 회귀(확률로 답하는 모델)":
    selected_predictions = logistic_test_pred
    selected_cm = logistic_cm
else:
    selected_predictions = tree_test_pred
    selected_cm = tree_cm


# 테스트 데이터에 예측 결과 붙이기
result_df = test_df[
    [
        "id",
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease",
        "stroke",
    ]
].copy()

result_df["예측"] = selected_predictions

# FN: 실제 뇌졸중인데 뇌졸중 아님으로 예측
missed_patients = result_df[
    (result_df["stroke"] == 1)
    & (result_df["예측"] == 0)
].copy()

# FP: 실제 뇌졸중은 아닌데 뇌졸중으로 예측
false_alarm_patients = result_df[
    (result_df["stroke"] == 0)
    & (result_df["예측"] == 1)
].copy()


def make_patient_table(data):
    table = data[
        [
            "id",
            "age",
            "avg_glucose_level",
            "hypertension",
            "heart_disease",
        ]
    ].copy()

    table = table.rename(columns={
        "id": "번호",
        "age": "나이",
        "avg_glucose_level": "평균 혈당",
        "hypertension": "고혈압 여부",
        "heart_disease": "심장병 여부",
    })

    table["고혈압 여부"] = table["고혈압 여부"].map({
        0: "아님",
        1: "있음",
    })

    table["심장병 여부"] = table["심장병 여부"].map({
        0: "아님",
        1: "있음",
    })

    table = table.sort_values(
        "나이",
        ascending=False,
    )

    return table


missed_table = make_patient_table(missed_patients)
false_alarm_table = make_patient_table(false_alarm_patients)

list_col1, list_col2 = st.columns(2)

with list_col1:
    st.markdown(
        f"**놓친 환자: {len(missed_table):,}명**"
    )
    st.caption(
        "실제 뇌졸중 환자였지만 모델이 뇌졸중 아님으로 예측한 사람(FN)"
    )

    st.dataframe(
        missed_table.head(20),
        use_container_width=True,
        hide_index=True,
    )

    if len(missed_table) > 20:
        st.caption(
            f"전체 {len(missed_table):,}명 중 나이가 많은 앞 20명만 표시했습니다."
        )


with list_col2:
    st.markdown(
        f"**헛짚은 사람: {len(false_alarm_table):,}명**"
    )
    st.caption(
        "실제 뇌졸중은 아니지만 모델이 뇌졸중이라고 예측한 사람(FP)"
    )

    st.dataframe(
        false_alarm_table.head(20),
        use_container_width=True,
        hide_index=True,
    )

    if len(false_alarm_table) > 20:
        st.caption(
            f"전체 {len(false_alarm_table):,}명 중 나이가 많은 앞 20명만 표시했습니다."
        )


# --------------------------------------------------
# 목록 수와 혼동행렬 확인
# --------------------------------------------------
tn, fp, fn, tp = selected_cm.ravel()

st.info(
    f"놓친 환자 목록은 혼동행렬의 FN과 같은 **{fn:,}명**이고, "
    f"헛짚은 사람 목록은 FP와 같은 **{fp:,}명**입니다."
)
