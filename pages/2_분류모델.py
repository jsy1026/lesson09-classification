import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="분류 모델 | 뇌졸중 예측 실습실",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 분류 모델")
st.write(
    "입력 속성을 골라 로지스틱 회귀와 의사결정트리 모델을 학습하고, "
    "테스트 데이터로 예측 결과를 살펴봅니다."
)

DATA_URL = "https://raw.githubusercontent.com/jsy1026/data/refs/heads/main/stroke.csv"

FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}

ALL_FEATURES = list(FEATURE_NAMES.keys())


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    needed_columns = [
        "id",
        "age",
        "avg_glucose_level",
        "bmi",
        "hypertension",
        "heart_disease",
        "stroke",
    ]

    df = df[needed_columns].copy()

    for col in ALL_FEATURES + ["stroke"]:
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
# 입력 속성 선택
# --------------------------------------------------
st.header("1. 입력 속성 선택")

selected_features = st.multiselect(
    "모델의 입력으로 사용할 속성을 선택하세요.",
    options=ALL_FEATURES,
    default=[
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease",
    ],
    format_func=lambda feature: FEATURE_NAMES[feature],
)

if len(selected_features) < 2:
    st.warning("입력 속성을 두 개 이상 선택해 주세요.")
    st.stop()

st.write(
    "선택한 속성: "
    + ", ".join(FEATURE_NAMES[feature] for feature in selected_features)
)


# --------------------------------------------------
# 데이터 나누기
# 번호 순으로 10명씩 묶고, 각 묶음 앞 3명은 테스트용
# --------------------------------------------------
df["묶음 번호"] = df.index // 10
df["묶음 안 순서"] = df.index % 10

test_df = df[df["묶음 안 순서"] < 3].copy()
train_df = df[df["묶음 안 순서"] >= 3].copy()

# 선택한 속성에 결측치가 있는 행은 제외
train_df = train_df.dropna(
    subset=selected_features + ["stroke"]
).copy()

test_df = test_df.dropna(
    subset=selected_features + ["stroke"]
).copy()


# --------------------------------------------------
# BMI를 선택한 경우에만 학습용 중앙값으로 결측치 채우기
# --------------------------------------------------
if "bmi" in selected_features:
    bmi_median = train_df["bmi"].median()

    train_df["bmi"] = train_df["bmi"].fillna(bmi_median)
    test_df["bmi"] = test_df["bmi"].fillna(bmi_median)

    st.caption(
        f"체질량지수의 빈 값은 학습용 데이터의 중앙값 "
        f"{bmi_median:.2f}으로 채웠습니다."
    )


# --------------------------------------------------
# 학습용 / 테스트용 데이터 준비
# --------------------------------------------------
X_train = train_df[selected_features]
y_train = train_df["stroke"].astype(int)

X_test = test_df[selected_features]
y_test = test_df["stroke"].astype(int)

st.write(
    f"**학습용: {len(train_df):,}명 | "
    f"테스트용: {len(test_df):,}명**"
)

if len(test_df) != 1533:
    st.warning(
        "테스트용 데이터가 1,533명이 아닙니다. "
        "선택한 속성의 결측치 때문에 일부 행이 제외되었는지 확인해 주세요."
    )


# --------------------------------------------------
# 모델 학습
# --------------------------------------------------
# 로지스틱 회귀: 크기 맞추기를 위해 StandardScaler 사용
logistic_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=2000,
        random_state=42,
    ),
)

logistic_model.fit(X_train, y_train)


# 의사결정트리: 질문 최대 3번, 마지막 마디 최소 5명
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42,
)

tree_model.fit(X_train, y_train)


# --------------------------------------------------
# 기준 모델
# 입력을 보지 않고 학습용에서 많은 쪽으로만 답하기
# --------------------------------------------------
majority_class = int(y_train.value_counts().idxmax())

baseline_train_pred = np.full(
    len(y_train),
    majority_class,
)

baseline_test_pred = np.full(
    len(y_test),
    majority_class,
)


# --------------------------------------------------
# 정확도 계산
# --------------------------------------------------
logistic_train_pred = logistic_model.predict(X_train)
logistic_test_pred = logistic_model.predict(X_test)

tree_train_pred = tree_model.predict(X_train)
tree_test_pred = tree_model.predict(X_test)

logistic_train_accuracy = accuracy_score(
    y_train,
    logistic_train_pred,
)

logistic_test_accuracy = accuracy_score(
    y_test,
    logistic_test_pred,
)

tree_train_accuracy = accuracy_score(
    y_train,
    tree_train_pred,
)

tree_test_accuracy = accuracy_score(
    y_test,
    tree_test_pred,
)

baseline_train_accuracy = accuracy_score(
    y_train,
    baseline_train_pred,
)

baseline_test_accuracy = accuracy_score(
    y_test,
    baseline_test_pred,
)


# --------------------------------------------------
# 정확도 카드
# --------------------------------------------------
st.header("2. 모델 정확도")

card1, card2, card3 = st.columns(3)

with card1:
    st.metric(
        label="로지스틱 회귀(확률로 답하는 모델)",
        value=f"{logistic_test_accuracy * 100:.2f}%",
        help="큰 숫자는 테스트 데이터 정확도입니다.",
    )
    st.caption(
        f"훈련 데이터 정확도: {logistic_train_accuracy * 100:.2f}%  \n"
        f"테스트 데이터 정확도: {logistic_test_accuracy * 100:.2f}%"
    )

with card2:
    st.metric(
        label="의사결정트리(질문으로 답하는 모델)",
        value=f"{tree_test_accuracy * 100:.2f}%",
        help="큰 숫자는 테스트 데이터 정확도입니다.",
    )
    st.caption(
        f"훈련 데이터 정확도: {tree_train_accuracy * 100:.2f}%  \n"
        f"테스트 데이터 정확도: {tree_test_accuracy * 100:.2f}%"
    )

with card3:
    st.metric(
        label="기준 모델(입력을 보지 않고 많은 쪽으로 답함)",
        value=f"{baseline_test_accuracy * 100:.2f}%",
        help="학습용에서 더 많은 쪽의 답만 계속 예측합니다.",
    )
    st.caption(
        f"훈련 데이터 정확도: {baseline_train_accuracy * 100:.2f}%  \n"
        f"테스트 데이터 정확도: {baseline_test_accuracy * 100:.2f}%"
    )


# --------------------------------------------------
# 그림의 가로축 / 세로축 선택
# --------------------------------------------------
st.header("3. 두 속성으로 살펴보기")

axis_col1, axis_col2 = st.columns(2)

with axis_col1:
    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        index=0,
        format_func=lambda feature: FEATURE_NAMES[feature],
    )

with axis_col2:
    y_options = [
        feature
        for feature in selected_features
        if feature != x_feature
    ]

    y_feature = st.selectbox(
        "세로축 속성",
        options=y_options,
        index=0,
        format_func=lambda feature: FEATURE_NAMES[feature],
    )


# --------------------------------------------------
# 두 축 이외의 속성은 테스트 데이터 중앙값에 고정
# --------------------------------------------------
fixed_values = {}

for feature in selected_features:
    if feature not in [x_feature, y_feature]:
        fixed_values[feature] = float(
            test_df[feature].median()
        )

if fixed_values:
    fixed_text = ", ".join(
        f"{FEATURE_NAMES[feature]} = {value:.2f}"
        for feature, value in fixed_values.items()
    )

    st.write(
        "그림의 두 축이 아닌 속성은 테스트 데이터의 중앙값에 고정했습니다: "
        + fixed_text
    )
else:
    st.write("선택한 속성이 두 개뿐이므로 고정한 속성은 없습니다.")


# --------------------------------------------------
# 그림을 위한 격자 데이터 만들기
# --------------------------------------------------
x_min = float(test_df[x_feature].min())
x_max = float(test_df[x_feature].max())

y_min = float(test_df[y_feature].min())
y_max = float(test_df[y_feature].max())

x_padding = (x_max - x_min) * 0.05
y_padding = (y_max - y_min) * 0.05

if x_padding == 0:
    x_padding = 1

if y_padding == 0:
    y_padding = 1

x_range = np.linspace(
    x_min - x_padding,
    x_max + x_padding,
    100,
)

y_range = np.linspace(
    y_min - y_padding,
    y_max + y_padding,
    100,
)

xx, yy = np.meshgrid(
    x_range,
    y_range,
)

grid_df = pd.DataFrame(
    np.tile(
        X_test.median().to_numpy(),
        (xx.size, 1),
    ),
    columns=selected_features,
)

grid_df[x_feature] = xx.ravel()
grid_df[y_feature] = yy.ravel()


# --------------------------------------------------
# 로지스틱 회귀의 뇌졸중 확률 및 0.5 경계
# --------------------------------------------------
logistic_probabilities = logistic_model.predict_proba(
    grid_df
)[:, 1]

logistic_probability_grid = logistic_probabilities.reshape(
    xx.shape
)


# --------------------------------------------------
# 의사결정트리의 영역
# --------------------------------------------------
tree_grid_predictions = tree_model.predict(
    grid_df
)

tree_region_grid = tree_grid_predictions.reshape(
    xx.shape
)


# --------------------------------------------------
# Plotly 그림
# --------------------------------------------------
fig = go.Figure()


# 의사결정트리 영역을 옅은 색으로 표시
fig.add_trace(
    go.Contour(
        x=x_range,
        y=y_range,
        z=tree_region_grid,
        showscale=False,
        opacity=0.18,
        colorscale=[
            [0.0, "#4C78A8"],
            [1.0, "#E45756"],
        ],
        contours={
            "start": 0,
            "end": 1,
            "size": 1,
            "coloring": "fill",
            "showlines": False,
        },
        hoverinfo="skip",
        name="의사결정트리 영역",
    )
)


# 로지스틱 회귀의 0.5 경계선
fig.add_trace(
    go.Contour(
        x=x_range,
        y=y_range,
        z=logistic_probability_grid,
        showscale=False,
        contours={
            "start": 0.5,
            "end": 0.5,
            "size": 1,
            "coloring": "lines",
            "showlabels": False,
        },
        line={
            "color": "black",
            "width": 3,
        },
        hoverinfo="skip",
        name="로지스틱 회귀 0.5 경계",
    )
)


# 테스트 데이터 점: 실제 뇌졸중 여부로 색 구분
plot_df = test_df.copy()

plot_df["실제 뇌졸중 여부"] = plot_df["stroke"].map({
    0: "뇌졸중 아님",
    1: "뇌졸중",
})

fig.add_trace(
    go.Scatter(
        x=plot_df[x_feature],
        y=plot_df[y_feature],
        mode="markers",
        marker={
            "size": 7,
            "opacity": 0.7,
        },
        name="테스트 데이터",
        text=plot_df["실제 뇌졸중 여부"],
        customdata=plot_df[["id"]],
        hovertemplate=(
            "번호: %{customdata[0]}<br>"
            + FEATURE_NAMES[x_feature]
            + ": %{x}<br>"
            + FEATURE_NAMES[y_feature]
            + ": %{y}<br>"
            + "실제 결과: %{text}<extra></extra>"
        ),
    )
)


fig.update_layout(
    title="테스트 데이터와 두 모델의 판단 영역",
    xaxis_title=FEATURE_NAMES[x_feature],
    yaxis_title=FEATURE_NAMES[y_feature],
    legend_title="범례",
    height=650,
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# --------------------------------------------------
# 로지스틱 회귀 경계선이 그림 범위 안에 있는지 안내
# --------------------------------------------------
boundary_points = []

for i in range(len(x_range)):
    for j in range(len(y_range) - 1):
        p1 = logistic_probability_grid[j, i]
        p2 = logistic_probability_grid[j + 1, i]

        if (p1 - 0.5) * (p2 - 0.5) <= 0:
            boundary_points.append(
                (
                    x_range[i],
                    (y_range[j] + y_range[j + 1]) / 2,
                )
            )
            break

if boundary_points:
    st.caption(
        "로지스틱 회귀의 0.5 경계선이 그림 범위 안에 있습니다."
    )
else:
    st.caption(
        "로지스틱 회귀의 0.5 경계선은 그림 범위 안에서 발견되지 않았습니다. "
        "경계선이 그림 밖에 있거나, 이 범위에서 확률이 0.5를 지나지 않을 수 있습니다."
    )


# --------------------------------------------------
# 의사결정트리 가지 그림 만들기
# --------------------------------------------------
st.header("4. 의사결정트리가 던진 질문")

tree_structure = tree_model.tree_
feature_names = [
    FEATURE_NAMES[feature]
    for feature in selected_features
]

dot_lines = [
    "digraph DecisionTree {",
    'graph [rankdir=TB, bgcolor="white"];',
    'node [shape=box, style="rounded,filled", fontname="Arial"];',
    'edge [fontname="Arial", fontsize=10];',
]


def build_dot(node_id):
    left_child = tree_structure.children_left[node_id]
    right_child = tree_structure.children_right[node_id]

    node_samples = int(tree_structure.n_node_samples[node_id])
    node_values = tree_structure.value[node_id][0]

    negative_count = int(round(node_values[0]))
    positive_count = int(round(node_values[1]))

    positive_rate = (
        positive_count / node_samples
        if node_samples > 0
        else 0
    )

    is_leaf = (
        left_child == _tree.TREE_LEAF
        and right_child == _tree.TREE_LEAF
    )

    if is_leaf:
        predicted_class = int(tree_structure.value[node_id][0].argmax())

        if predicted_class == 1:
            fill_color = "#F8CACA"
            answer_text = "답: 뇌졸중"
        else:
            fill_color = "#CFE3F5"
            answer_text = "답: 뇌졸중 아님"

        label = (
            f"마디 {node_id}\\n"
            f"{answer_text}\\n"
            f"사람 수: {node_samples}명\\n"
            f"뇌졸중: {positive_count}명\\n"
            f"비율: {positive_rate * 100:.1f}%"
        )

        dot_lines.append(
            f'{node_id} [label="{label}", fillcolor="{fill_color}"];'
        )

        return

    feature_index = tree_structure.feature[node_id]
    threshold = tree_structure.threshold[node_id]
    feature_name = feature_names[feature_index]

    label = (
        f"마디 {node_id}\\n"
        f"{feature_name} ≤ {threshold:.2f}?\\n"
        f"사람 수: {node_samples}명\\n"
        f"뇌졸중: {positive_count}명\\n"
        f"비율: {positive_rate * 100:.1f}%"
    )

    dot_lines.append(
        f'{node_id} [label="{label}", fillcolor="#FFF2CC"];'
    )

    dot_lines.append(
        f'{node_id} -> {left_child} [label="예"];'
    )

    dot_lines.append(
        f'{node_id} -> {right_child} [label="아니요"];'
    )

    build_dot(left_child)
    build_dot(right_child)


build_dot(0)

dot_lines.append("}")

dot_string = "\n".join(dot_lines)

st.graphviz_chart(
    dot_string,
    use_container_width=True,
)


# --------------------------------------------------
# 트리 요약
# --------------------------------------------------
leaf_nodes = np.where(
    tree_structure.children_left == _tree.TREE_LEAF
)[0]

negative_leaf_count = 0

for leaf_id in leaf_nodes:
    predicted_class = int(
        tree_structure.value[leaf_id][0].argmax()
    )

    if predicted_class == 0:
        negative_leaf_count += 1

used_feature_indices = sorted(
    set(
        tree_structure.feature[
            tree_structure.feature >= 0
        ]
    )
)

used_features = [
    selected_features[index]
    for index in used_feature_indices
]

if used_features:
    used_feature_text = ", ".join(
        FEATURE_NAMES[feature]
        for feature in used_features
    )
else:
    used_feature_text = "없음"


st.write(
    f"**답을 내는 마디 수:** {len(leaf_nodes)}칸, "
    f"그중 뇌졸중 아님으로 답하는 마디는 {negative_leaf_count}칸입니다."
)

st.write(
    f"**이 나무가 실제로 물어본 속성:** {used_feature_text}"
)


# --------------------------------------------------
# 안내
# --------------------------------------------------
st.caption(
    "참고: 정확도는 테스트 데이터에서 맞힌 비율입니다. "
    "모델의 예측 결과만으로 뇌졸중의 원인이나 개인의 실제 위험을 확정할 수는 없습니다."
)
