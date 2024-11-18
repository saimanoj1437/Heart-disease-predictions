import streamlit as st
import pandas as pd
import pickle

# Page Configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Styling
st.markdown(
    """
    <style>
    body {
        background-color: #f7f7f7;
    }
    .main-title {
        font-size: 2.5em;
        font-weight: 700;
        text-align: center;
        color: #dc3545;
    }
    .sub-title {
        font-size: 1.2em;
        text-align: center;
        color: #333;
        margin-bottom: 40px;
    }
    .result {
        font-size: 1.5em;
        font-weight: bold;
        text-align: center;
        margin-top: 30px;
        padding: 15px;
        border-radius: 10px;
    }
    .positive {
        background-color: #f8d7da;
        color: #721c24;
    }
    .negative {
        background-color: #d4edda;
        color: #155724;
    }
    .footer {
        margin-top: 50px;
        text-align: center;
        color: #555;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title Section
st.markdown('<div class="main-title">Heart Disease Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Enter the details below to check for potential signs of heart disease. This tool uses a Random Forest Classifier for prediction.</div>',
    unsafe_allow_html=True,
)

# Load the model
pred = pickle.load(open('rf.pkl', 'rb'))

# Input Section
def user_input_features():
    with st.sidebar:
        st.header("Input Your Details:")
        age = st.number_input('Age:', value=25, min_value=1, max_value=100)
        sex = st.radio('Sex:', [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
        cp = st.selectbox('Chest Pain Type:', [0, 1, 2, 3])
        tres = st.number_input('Resting Blood Pressure:', value=120, min_value=0, max_value=300)
        chol = st.number_input('Serum Cholesterol (mg/dl):', value=200, min_value=0, max_value=600)
        fbs = st.radio('Fasting Blood Sugar > 120 mg/dl:', [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        res = st.selectbox('Resting ECG Results:', [0, 1, 2])
        tha = st.number_input('Max Heart Rate Achieved:', value=150, min_value=0, max_value=300)
        exa = st.radio('Exercise-Induced Angina:', [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        old = st.slider('ST Depression (Oldpeak):', 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox('Slope of ST Segment:', [0, 1, 2])
        ca = st.selectbox('Number of Major Vessels:', [0, 1, 2, 3])
        thal = st.selectbox('Thalassemia:', [1, 2, 3])

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': tres,
        'chol': chol,
        'fbs': fbs,
        'restecg': res,
        'thalach': tha,
        'exang': exa,
        'oldpeak': old,
        'slope': slope,
        'ca': ca,
        'thal': thal,
    }
    input_df = pd.DataFrame(data, index=[0])
    return input_df

# Get User Input
input_df = user_input_features()

# Load Dataset for Dummy Columns
dataset = pd.read_csv('heart_data.csv')
dataset = dataset.drop(columns=['target'])

# Prepare DataFrame
df = pd.concat([input_df, dataset], axis=0)
df = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
df = df[:1]  # Use only user data for prediction

# Predict Button
if st.button('Predict ❤️'):
    prediction = pred.predict(df)

    if prediction[0] == 1:
        st.markdown(
            '<div class="result positive">Positive for Heart Disease 🚨</div>',
            unsafe_allow_html=True,
        )
        st.write(
            "⚠️ It's strongly recommended to consult a cardiologist for further evaluation and appropriate medical advice."
        )
    else:
        st.markdown(
            '<div class="result negative">Negative for Heart Disease ✅</div>',
            unsafe_allow_html=True,
        )
        st.write(
            "✨ Great! Maintain a healthy lifestyle to keep your heart in good shape. Regular check-ups are still important."
        )

# Footer
st.markdown('<div class="footer">A PROJECT BY - M. MANOJ BHASKAR</div>', unsafe_allow_html=True)
