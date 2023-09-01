import streamlit as st
import pandas as pd
import pickle


st.set_page_config(page_title="Heart Disease Prediction")

st.title('Heart Disease Prediction')


pred = pickle.load(open('rf.pkl','rb'))


def user_input_features():
    age = st.number_input('Enter your age: ', value=25, min_value=1, max_value=100)

    sex = st.selectbox('Sex:  [0 - MALE | 1 - FEMALE)', [0, 1])

    cp = st.selectbox('Chest pain type', [0, 1, 2, 3])

    tres = st.number_input('Resting blood pressure: ', value=120, min_value=0, max_value=300)

    chol = st.number_input('Serum cholestoral in mg/dl: ', value=200, min_value=0, max_value=600)

    fbs = st.selectbox('Fasting blood sugar:  [0 - NO | 1 - YES)', [0, 1])

    res = st.number_input('Resting electrocardiographic results: ', value=0, min_value=0, max_value=2)

    tha = st.number_input('Maximum heart rate achieved: ', value=150, min_value=0, max_value=300)

    exa = st.selectbox('Exercise induced angina:  [0 - NO | 1 - YES)', [0, 1])

    old = st.number_input('Oldpeak: ', value=1.0, min_value=0.0, max_value=10.0, step=0.1)

    slope = st.number_input('Slope of the peak exercise ST segment: ', value=1, min_value=0, max_value=2)

    ca = st.selectbox('Number of major vessels', [0, 1, 2, 3])

    thal = st.selectbox('Thal', [0, 1, 2])

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
        'thal': thal
    }
    input_df = pd.DataFrame(data, index=[0])
    return input_df

# Get user input
input_df = user_input_features()

# Load dataset
dataset = pd.read_csv('heart_data.csv')
dataset = dataset.drop(columns=['target'])

df = pd.concat([input_df, dataset], axis=0)

# Perform one-hot encoding
df = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

df = df[:1]

# Create a button to trigger prediction
if st.button('Predict'):

    prediction = pred.predict(df)

    if prediction[0] == 1:

        st.write('Positive for Heart Disease')
        
    else:
        st.write('Negative for Heart Disease')


st.markdown("A PROJECT BY - M.MANOJ BHASKAR")

