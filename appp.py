import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Heart Disease Prediction")

st.title('Heart Disease Prediction')

st.write("This model attempts to predict whether or not someone has heart disease using the XGB CLASSIFIER.\n\nThe model was trained with 100 percent accuracy\n\nPlease note that this should not be used as a medical diagnosis, rather just a tool to help. Please consult your local medical professional if you have any health concerns.",width=500)
st.markdown("please enter the information below")


pred = pickle.load(open('rf.pkl','rb'))


def user_input_features():
    age = st.slider('Enter your age: ', value=25, min_value=1, max_value=100)

    sex = st.selectbox('Sex:  [0 - MALE | 1 - FEMALE)', [0, 1])

    cp = st.selectbox('Chest pain type', [0, 1, 2, 3])

    tres = st.slider('Resting blood pressure: ', value=120, min_value=0, max_value=300)

    chol = st.slider('Serum cholestoral in mg/dl: ', value=200, min_value=0, max_value=600)

    fbs = st.selectbox('Fasting blood sugar:  [0 - NO | 1 - YES)', [0, 1])

    res = st.slider('Resting electrocardiographic results: ', value=0, min_value=0, max_value=2)

    tha = st.slider('Maximum heart rate achieved: ', value=150, min_value=0, max_value=300)

    exa = st.selectbox('Exercise induced angina:  [0 - NO | 1 - YES)', [0, 1])

    old = st.slider('Oldpeak: ', value=1.0, min_value=0.0, max_value=10.0, step=0.1)

    slope = st.slider('Slope of the peak exercise ST segment: ', value=1, min_value=0, max_value=2)

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

st.sidebar.markdown("Dataset Features\n\n\nYou can view the entire Jupyter Notebook on Github (click the three bars on the top right of your screen and select 'View app source')\n\nage - Patient age.\n\nsex - Patient sex assigned at birth. (1 = male, 0 = female).\n\ncp - Chest pain type. (0 = typical angina, 1 = atypical angina, 2 = non—anginal pain, 3 = asymptotic).\n\ntrestbps - Resting blood pressure (mmHg).\n\nchol - Serum Cholesterol level (mg/dl).\n\nfbs - Fasting Blood Sugar level (mg/dl). (1 = fasting blood sugar is more than 120mg/dl, 0 = other).\n\nrestecg - Resting ElectroCardioGraphic results (0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hyperthrophy).\n\nthalach - Max heart rate achieved.\n\nexang - Exercise induced angina (1 = yes, 0 = no).\n\noldpeak - ST depression induced by exercise relative to rest.\n\nslope - Peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping).\n\nca - Number of major vessels (0–3) colored by flourosopy.\n\nthal - Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect).\n\ntarget - Heart disease prediction (0 = absence, 1 = present)")


st.markdown("A PROJECT BY - M.MANOJ BHASKAR")

