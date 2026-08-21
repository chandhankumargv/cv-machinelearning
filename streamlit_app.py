import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
# TITLE
st.title("PREDICTION APP")

st.info("APP WHICH USES ML MODELS FOR PREDICTION")

# LOAD DATA
df = pd.read_csv(
    "https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv"
)

# DATA

with st.expander("Data"):

    st.write("### Raw Data")
    st.dataframe(df)

    st.write("### X")
    x = df.drop("species", axis=1)
    st.dataframe(x)

    st.write("### Y")
    y = df["species"]
    st.dataframe(y)

# DATA VISUALISATION
with st.expander("Data Visualisation"):

    st.scatter_chart(
        data=df,
        x="bill_length_mm",
        y="body_mass_g",
        color="species"
    )

# SIDEBAR
with st.sidebar:

    st.header("Input Features")

    island = st.selectbox(
        "Island",
        ["Torgersen", "Biscoe", "Dream"]
    )

    sex = st.selectbox(
        "Sex",
        ["male", "female"]
    )

    bill_length_mm = st.slider(
        "Bill Length (mm)",
        32.1,
        59.6,
        43.9
    )

    bill_depth_mm = st.slider(
        "Bill Depth (mm)",
        13.1,
        21.5,
        17.2
    )

    flipper_length_mm = st.slider(
        "Flipper Length (mm)",
        172.0,
        231.0,
        201.0
    )

    body_mass_g = st.slider(
        "Body Mass (g)",
        2700.0,
        6300.0,
        4207.0
    )

# INPUT DATAFRAME

data = {
    "island": [island],
    "bill_length_mm": [bill_length_mm],
    "bill_depth_mm": [bill_depth_mm],
    "flipper_length_mm": [flipper_length_mm],
    "body_mass_g": [body_mass_g],
    "sex": [sex]
}

input_df = pd.DataFrame(data)

# DISPLAY INPUT

with st.expander("Input Features"):

    st.write("### Input Penguin")

    st.dataframe(input_df)

# FEATURES
categorical_features = [
    "island",
    "sex"
]

numerical_features = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
]

# PREPROCESSING

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

# MODEL

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=1000)
        )
    ]
)

# TRAIN MODEL

model.fit(x, y)

# PREDICTION

prediction = model.predict(input_df)

probability = model.predict_proba(input_df)

# RESULT
st.subheader("Prediction")

st.success(
    f"Predicted Penguin Species: {prediction[0]}"
)


# PROBABILITY

st.subheader("Prediction Probability")

probability_df = pd.DataFrame(
    probability,
    columns=model.classes_
)

st.dataframe(probability_df)


# --------------------------------------------------
# PROBABILITY CHART
# --------------------------------------------------

st.bar_chart(probability_df.T)
