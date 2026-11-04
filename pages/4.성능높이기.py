import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="성능 높이기",
    page_icon="⚖️",
    layout="wide",
)

DATA_URL = "https://raw.githubusercontent.com/jsy1026/data/refs/heads/main/stroke.csv"

FEATURE_LABELS = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}

DEFAULT_FEATURES = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease",
]

MODEL_LABELS = {
    "로지스틱 회귀": "로지스틱 회귀(확률로 답하는 모델)",
    "의사결정트리": "의사결정트리(질문으로 답하는 모델)",
}


# -----------------------------
# 데이터 불러오기
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    data = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.exception(e)
    st.stop()


# -----------------------------
# 페이지 제목
# -----------------------------
st.title("⚖️ 성능 높이기")

st.write(
    "뇌졸중 환자가 드문 데이터에서, "
    "두 클래스의 가중치를 맞추기 전과 후를 비교합니다."
)


# -----------------------------
# 입력 속성 선택
# -----------------------------
selected_features = st.multiselect(
    "모델 입력으로 사용할 속성을 선택하세요.",
    options=list(FEATURE_LABELS.keys()),
    default=DEFAULT_FEATURES,
    format_func=lambda x: FEATURE_LABELS[x],
)

if len(selected_features) < 2:
    st.warning("비교와 그래프를 위해 속성을 두 개 이상 선택해 주세요.")
    st.stop()


# -----------------------------
# 데이터 정리
# -----------------------------
required_columns = ["id", "stroke"] + selected_features

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    st.error(
        "데이터에 필요한 열이 없습니다: "
        + ", ".join(missing_columns)
    )
    st.stop()


df = data[required_columns].copy()

df["id"] = pd.to_numeric(df["id"], errors="coerce")
df["stroke"] = pd.to_numeric(df["stroke"], errors="coerce")

for feature in selected_features:
    df[feature] = pd.to_numeric(df[feature], errors="coerce")


df = (
    df.dropna(subset=["id", "stroke"])
    .sort_values("id")
    .reset_index(drop=True)
)


# -----------------------------
# 채점표 페이지와 같은 방식으로 데이터 나누기
# id 정렬 후 10명씩 묶고, 각 묶음의 앞 3명을 테스트 데이터로 사용
# -----------------------------
df["_순서"] = np.arange(len(df))
df["_묶음순서"] = df["_순서"] % 10

test_df = df[df["_묶음순서"] < 3].copy()
train_df = df[df["_묶음순서"] >= 3].copy()


# -----------------------------
# BMI 결측값 처리
# 훈련 데이터의 중앙값만 사용
# -----------------------------
if "bmi" in selected_features:
    bmi_median = train_df["bmi"].median()

    if pd.isna(bmi_median):
        st.error("훈련 데이터의 BMI 중앙값을 계산할 수 없습니다.")
        st.stop()

    train_df["bmi"] = train_df["bmi"].fillna(bmi_median)
    test_df["bmi"] = test_df["bmi"].fillna(bmi_median)

    st.caption(
        f"체질량지수 결측값은 훈련 데이터 중앙값 "
        f"({bmi_median:.2f})으로 채웠습니다."
    )


# 선택 속성 중 BMI 이외의 결측 행이 있으면 제외
train_df = train_df.dropna(
    subset=selected_features + ["stroke"]
).copy()

test_df = test_df.dropna(
    subset=selected_features + ["stroke"]
).copy()


if train_df.empty or test_df.empty:
    st.error("결측값 처리 후 훈련 또는 테스트 데이터가 비어 있습니다.")
    st.stop()


X_train = train_df[selected_features].astype(float)
y_train = train_df["stroke"].astype(int)

X_test = test_df[selected_features].astype(float)
y_test = test_df["stroke"].astype(int)


if y_train.nunique() < 2:
    st.error(
        "훈련 데이터에 뇌졸중 여부가 한 종류만 있어 "
        "모델을 학습할 수 없습니다."
    )
    st.stop()


st.caption(
    f"훈련 데이터: {len(train_df):,}명 · "
    f"테스트 데이터: {len(test_df):,}명"
)


# -----------------------------
# 모델 만들기
# -----------------------------
models_before = {
    "로지스틱 회귀": Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),
        ),
    ]),
    "의사결정트리": DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=5,
        random_state=42,
    ),
}


models_after = {
    "로지스틱 회귀": Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42,
                class_weight="balanced",
            ),
        ),
    ]),
    "의사결정트리": DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
    ),
}


# -----------------------------
# 모델 학습 및 예측
# -----------------------------
results = {}

for model_key in models_before:
    model_before = models_before[model_key]
    model_after = models_after[model_key]

    model_before.fit(X_train, y_train)
    model_after.fit(X_train, y_train)

    pred_before = model_before.predict(X_test)
    pred_after = model_after.predict(X_test)

    results[model_key] = {
        "before_model": model_before,
        "after_model": model_after,
        "before_pred": pred_before,
        "after_pred": pred_after,
    }


# -----------------------------
# 성능 지표 계산 함수
# -----------------------------
def get_metrics(y_true, y_pred):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "정확도": accuracy_score(y_true, y_pred),
        "정밀도": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "재현율": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "F1 점수": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }


# -----------------------------
# 1. 가중치 적용 전후 결과 비교
# -----------------------------
st.header("1. 가중치 적용 전·후 결과 비교")


for model_key, model_result in results.items():

    st.subheader(MODEL_LABELS[model_key])

    before = get_metrics(
        y_test,
        model_result["before_pred"],
    )

    after = get_metrics(
        y_test,
        model_result["after_pred"],
    )

    comparison_table = pd.DataFrame({
        "항목": [
            "TN: 정상으로 맞힘",
            "FP: 정상인을 환자로 오판",
            "FN: 환자를 놓침",
            "TP: 환자를 찾아냄",
            "정확도",
            "정밀도",
            "재현율",
            "F1 점수",
        ],
        "가중치 적용 전": [
            f'{before["TN"]:,}',
            f'{before["FP"]:,}',
            f'{before["FN"]:,}',
            f'{before["TP"]:,}',
            f'{before["정확도"]:.3f}',
            f'{before["정밀도"]:.3f}',
            f'{before["재현율"]:.3f}',
            f'{before["F1 점수"]:.3f}',
        ],
        "가중치 적용 후": [
            f'{after["TN"]:,}',
            f'{after["FP"]:,}',
            f'{after["FN"]:,}',
            f'{after["TP"]:,}',
            f'{after["정확도"]:.3f}',
            f'{after["정밀도"]:.3f}',
            f'{after["재현율"]:.3f}',
            f'{after["F1 점수"]:.3f}',
        ],
    })

    st.dataframe(
        comparison_table,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        f"**찾아낸 환자:** {before['TP']:,}명 → {after['TP']:,}명　|　"
        f"**놓친 환자:** {before['FN']:,}명 → {after['FN']:,}명　|　"
        f"**헛짚은 사람:** {before['FP']:,}명 → {after['FP']:,}명"
    )


# -----------------------------
# 2. 결정 경계 시각화
# -----------------------------
st.header("2. 가중치 적용 전·후 결정 경계")


plot_features = st.multiselect(
    "그래프 축으로 사용할 속성을 선택하세요.",
    options=selected_features,
    default=selected_features[:min(3, len(selected_features))],
    format_func=lambda x: FEATURE_LABELS[x],
    key="performance_plot_features",
)


if len(plot_features) < 2:
    st.info("그래프를 그리려면 속성을 두 개 이상 선택해 주세요.")

else:

    plot_df = test_df.copy()

    plot_df["실제 결과"] = plot_df["stroke"].map({
        0: "뇌졸중 없음",
        1: "뇌졸중 있음",
    })


    # -----------------------------
    # 2D 산점도
    # -----------------------------
    if len(plot_features) == 2:

        x_feature, y_feature = plot_features

        model_choice = st.selectbox(
            "경계선을 표시할 모델",
            options=list(results.keys()),
            format_func=lambda x: MODEL_LABELS[x],
            key="boundary_model_choice",
        )

        model_result = results[model_choice]

        fig = px.scatter(
            plot_df,
            x=x_feature,
            y=y_feature,
            color="실제 결과",
            labels={
                x_feature: FEATURE_LABELS[x_feature],
                y_feature: FEATURE_LABELS[y_feature],
                "실제 결과": "실제 뇌졸중 여부",
            },
            title="테스트 데이터와 가중치 적용 전·후 0.5 경계선",
            opacity=0.7,
        )


        # 축 범위 설정
        x_min = float(X_test[x_feature].min())
        x_max = float(X_test[x_feature].max())

        y_min = float(X_test[y_feature].min())
        y_max = float(X_test[y_feature].max())

        x_padding = (x_max - x_min) * 0.05 or 1
        y_padding = (y_max - y_min) * 0.05 or 1

        x_grid = np.linspace(
            x_min - x_padding,
            x_max + x_padding,
            180,
        )

        y_grid = np.linspace(
            y_min - y_padding,
            y_max + y_padding,
            180,
        )

        xx, yy = np.meshgrid(x_grid, y_grid)


        # 경계 계산용 입력 데이터 만들기
        def make_grid(model):

            grid = pd.DataFrame(
                index=np.arange(xx.size),
                columns=selected_features,
                dtype=float,
            )

            for feature in selected_features:

                if feature == x_feature:
                    grid[feature] = xx.ravel()

                elif feature == y_feature:
                    grid[feature] = yy.ravel()

                else:
                    grid[feature] = float(
                        X_test[feature].median()
                    )

            probabilities = model.predict_proba(
                grid[selected_features]
            )[:, 1]

            return probabilities.reshape(xx.shape)


        z_before = make_grid(
            model_result["before_model"]
        )

        z_after = make_grid(
            model_result["after_model"]
        )


        # 가중치 적용 전 경계선
        fig.add_trace(
            go.Contour(
                x=x_grid,
                y=y_grid,
                z=z_before,
                contours=dict(
                    start=0.5,
                    end=0.5,
                    size=1,
                    coloring="none",
                    showlines=True,
                ),
                line=dict(
                    color="#d62728",
                    width=3,
                ),
                showscale=False,
                name="가중치 적용 전 0.5 경계",
                hoverinfo="skip",
            )
        )


        # 가중치 적용 후 경계선
        fig.add_trace(
            go.Contour(
                x=x_grid,
                y=y_grid,
                z=z_after,
                contours=dict(
                    start=0.5,
                    end=0.5,
                    size=1,
                    coloring="none",
                    showlines=True,
                ),
                line=dict(
                    color="#1f77b4",
                    width=3,
                    dash="dash",
                ),
                showscale=False,
                name="가중치 적용 후 0.5 경계",
                hoverinfo="skip",
            )
        )


        st.plotly_chart(
            fig,
            use_container_width=True,
        )


        # 경계선이 그래프 범위에 있는지 확인
        before_has_boundary = (
            np.nanmin(z_before) <= 0.5 <= np.nanmax(z_before)
        )

        after_has_boundary = (
            np.nanmin(z_after) <= 0.5 <= np.nanmax(z_after)
        )

        if not before_has_boundary:
            st.info(
                "가중치 적용 전 모델의 0.5 경계선은 "
                "현재 그림 범위 밖에 있습니다."
            )

        if not after_has_boundary:
            st.info(
                "가중치 적용 후 모델의 0.5 경계선은 "
                "현재 그림 범위 밖에 있습니다."
            )


    # -----------------------------
    # 3D 산점도
    # -----------------------------
    else:

        x_feature, y_feature, z_feature = plot_features[:3]

        model_choice = st.selectbox(
            "3차원 경계 평면을 표시할 모델",
            options=list(results.keys()),
            format_func=lambda x: MODEL_LABELS[x],
            key="boundary_3d_model_choice",
        )

        model_result = results[model_choice]


        fig3d = px.scatter_3d(
            plot_df,
            x=x_feature,
            y=y_feature,
            z=z_feature,
            color="실제 결과",
            labels={
                x_feature: FEATURE_LABELS[x_feature],
                y_feature: FEATURE_LABELS[y_feature],
                z_feature: FEATURE_LABELS[z_feature],
                "실제 결과": "실제 뇌졸중 여부",
            },
            title="테스트 데이터와 가중치 적용 전·후 0.5 경계 평면",
            opacity=0.75,
        )


        x_values = np.linspace(
            float(X_test[x_feature].min()),
            float(X_test[x_feature].max()),
            35,
        )

        y_values = np.linspace(
            float(X_test[y_feature].min()),
            float(X_test[y_feature].max()),
            35,
        )

        z_values = np.linspace(
            float(X_test[z_feature].min()),
            float(X_test[z_feature].max()),
            45,
        )


        # 3D 결정 경계면을 근사적으로 계산
        def extract_boundary(model):

            boundary_x = []
            boundary_y = []
            boundary_z = []

            for xv in x_values:

                for yv in y_values:

                    grid = pd.DataFrame(
                        [{
                            feature: (
                                xv if feature == x_feature
                                else yv if feature == y_feature
                                else float(X_test[feature].median())
                            )
                            for feature in selected_features
                        }]
                    )

                    probabilities = []

                    for zv in z_values:

                        grid[z_feature] = zv

                        probability = model.predict_proba(
                            grid[selected_features]
                        )[0, 1]

                        probabilities.append(
                            float(probability)
                        )


                    # 확률 0.5를 통과하는 구간 찾기
                    boundary_value = None

                    for i in range(len(z_values) - 1):

                        p1 = probabilities[i]
                        p2 = probabilities[i + 1]

                        if (p1 - 0.5) * (p2 - 0.5) <= 0:

                            if p2 != p1:

                                boundary_value = (
                                    z_values[i]
                                    + (0.5 - p1)
                                    * (z_values[i + 1] - z_values[i])
                                    / (p2 - p1)
                                )

                            else:
                                boundary_value = z_values[i]

                            break


                    if boundary_value is not None:

                        boundary_x.append(xv)
                        boundary_y.append(yv)
                        boundary_z.append(boundary_value)


            return (
                boundary_x,
                boundary_y,
                boundary_z,
            )


        # 가중치 적용 전·후 경계면 추가
        boundary_models = [
            (
                model_result["before_model"],
                "#d62728",
                "가중치 적용 전 0.5 평면",
            ),
            (
                model_result["after_model"],
                "#1f77b4",
                "가중치 적용 후 0.5 평면",
            ),
        ]


        for model, color, name in boundary_models:

            bx, by, bz = extract_boundary(model)

            if len(bx) > 0:

                fig3d.add_trace(
                    go.Mesh3d(
                        x=bx,
                        y=by,
                        z=bz,
                        color=color,
                        opacity=0.35,
                        name=name,
                        alphahull=0,
                    )
                )

            else:

                st.info(
                    f"{name}은 현재 그래프 범위에서 "
                    "찾을 수 없습니다."
                )


        st.plotly_chart(
            fig3d,
            use_container_width=True,
        )

        st.caption(
            "3차원 산점도는 마우스로 회전하거나 확대해 볼 수 있습니다."
        )
