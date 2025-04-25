import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title='Heart Disease Prediction Dashboard', layout='wide')

# Custom CSS for professional medical analysis theme
st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');
.stApp {
    background-color: #ECF0F1;
    color: #2C3E50;
    font-family: 'Open Sans', sans-serif;
    animation: fadeIn 1s ease-in;
}
@keyframes fadeIn {
    0% { opacity: 0; }
    100% { opacity: 1; }
}
.css-1d391kg {
    background-color: #FFFFFF;
    border-right: 2px solid #4A90E2;
    position: sticky;
    top: 0;
    height: 100vh;
}
.css-1d391kg .stRadio > div {
    color: #4A90E2;
}
.css-1d391kg .stRadio > div label {
    font-weight: 600;
    transition: color 0.3s ease;
}
.css-1d391kg .stRadio > div label:hover {
    color: #2ECC71;
}
h1, h2, h3 {
    color: #4A90E2;
    display: flex;
    align-items: center;
}
h1::before, h2::before, h3::before {
    content: url('https://img.icons8.com/color/36/000000/heart-monitor.png');
    margin-right: 10px;
}
.card {
    background: #FFFFFF;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
    border: 1px solid #E0E0E0;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}
.card:hover {
    border-color: #2ECC71;
    box-shadow: 0 4px 15px rgba(46, 204, 113, 0.2);
    transform: translateY(-3px);
}
.stButton > button {
    background-color: #4A90E2;
    color: #FFFFFF;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background-color: #2ECC71;
    box-shadow: 0 0 10px rgba(46, 204, 113, 0.5);
    transform: scale(1.05);
}
.stNumberInput > div > div > input, .stSelectbox > div > div > select, .stTextInput > div > div > input {
    background-color: #F9FAFB;
    color: #2C3E50;
    border: 1px solid #4A90E2;
    border-radius: 5px;
}
.stNumberInput:hover > div > div > input, .stSelectbox:hover > div > div > select, .stTextInput:hover > div > div > input {
    border-color: #2ECC71;
}
.stMarkdown, .stDataFrame {
    color: #2C3E50;
}
.stDataFrame table {
    background-color: #FFFFFF;
    border: 1px solid #E0E0E0;
}
.stAlert {
    border-radius: 8px;
}
.stSuccess {
    background-color: #2ECC71;
    color: #FFFFFF;
}
.stWarning {
    background-color: #E74C3C;
    color: #FFFFFF;
}
</style>
''', unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('heart.csv')
        return df
    except FileNotFoundError:
        st.error("🚨 heart.csv not found in the repository. Please ensure it's in the root directory.")
        return None

# Preprocess data
@st.cache_data
def preprocess_data(df):
    df = df.copy()
    le = LabelEncoder()
    categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])
    return df, le

# Train models
@st.cache_resource
def train_models(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    rf.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    return rf, lr

# Sidebar navigation
st.sidebar.title('🩺 Heart Disease Prediction')
page = st.sidebar.radio('Navigate', ['What-If Analysis 🩺', 'Data Overview 📊', 'Scenario Manager ✅', 'Model Performance 📈'])

# Load data
df = load_data()
if df is None:
    st.stop()

# Preprocess data
df_processed, le = preprocess_data(df)
X = df_processed.drop('HeartDisease', axis=1)
y = df_processed['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
rf_model, lr_model = train_models(X_train, y_train)

# Initialize session state for scenarios
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = []

# What-If Analysis
if page == 'What-If Analysis 🩺':
    st.header('What-If Analysis 🩺')
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('Predict Heart Disease Risk ❤️')
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input('Age', min_value=20, max_value=100, value=50)
        sex = st.selectbox('Sex', ['M', 'F'])
        chest_pain = st.selectbox('Chest Pain Type', ['ATA', 'NAP', 'ASY', 'TA'])
        resting_bp = st.number_input('Resting BP', min_value=0, max_value=200, value=120)
        cholesterol = st.number_input('Cholesterol', min_value=0, max_value=600, value=200)
        fasting_bs = st.selectbox('Fasting BS', [0, 1])
    with col2:
        resting_ecg = st.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'])
        max_hr = st.number_input('Max HR', min_value=60, max_value=220, value=150)
        exercise_angina = st.selectbox('Exercise Angina', ['N', 'Y'])
        oldpeak = st.number_input('Oldpeak', min_value=-2.0, max_value=6.0, value=0.0, step=0.1)
        st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'])
        scenario_name = st.text_input('Scenario Name (Optional)', '')

    # Prepare input data
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })
    
    # Encode input data
    input_encoded = input_data.copy()
    categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    for col in categorical_columns:
        input_encoded[col] = le.fit_transform(input_encoded[col])
    
    # Prediction
    if st.button('Predict ❤️'):
        rf_pred = rf_model.predict(input_encoded)
        lr_pred = lr_model.predict(input_encoded)
        lr_prob = lr_model.predict_proba(input_encoded)[0][1]
        st.write(f'**Random Forest Prediction**: {"Heart Disease" if rf_pred[0] == 1 else "No Heart Disease"}')
        st.write(f'**Logistic Regression Prediction**: {"Heart Disease" if lr_pred[0] == 1 else "No Heart Disease"}')
        st.write(f'**Logistic Regression Probability**: {lr_prob:.2%}')
        if lr_prob >= 0.5:
            st.warning('🚨 High risk of heart disease detected!')
        else:
            st.success('✅ Low risk of heart disease.')

    # Save scenario
    if st.button('Save Scenario ✅') and scenario_name:
        st.session_state.scenarios.append({
            'name': scenario_name,
            'data': input_data.to_dict(),
            'rf_pred': rf_model.predict(input_encoded)[0],
            'lr_pred': lr_model.predict(input_encoded)[0],
            'lr_prob': lr_model.predict_proba(input_encoded)[0][1]
        })
        st.success(f'Scenario "{scenario_name}" saved!')

    # Sensitivity Analysis
    st.subheader('Sensitivity Analysis')
    feature = st.selectbox('Select Feature for Sensitivity Analysis', ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak'])
    if feature:
        range_values = {
            'Age': np.arange(20, 100, 5),
            'RestingBP': np.arange(80, 200, 10),
            'Cholesterol': np.arange(100, 400, 20),
            'MaxHR': np.arange(60, 220, 10),
            'Oldpeak': np.arange(-2, 6, 0.5)
        }
        probs = []
        for val in range_values[feature]:
            temp_data = input_encoded.copy()
            temp_data[feature] = val
            prob = lr_model.predict_proba(temp_data)[0][1]
            probs.append(prob)
        fig = px.line(x=range_values[feature], y=probs, labels={'x': feature, 'y': 'Probability of Heart Disease'})
        fig.update_layout(title=f'Sensitivity of {feature}', xaxis_title=feature, yaxis_title='Probability')
        st.plotly_chart(fig)

    # Logistic Regression Probability Analysis
    st.subheader('Logistic Regression Probability Analysis')
    st.write('Enter specific values to see how they affect the predicted probability of heart disease using the Logistic Regression model\'s fitted sigmoid function:')
    feature_values = {}
    for col in input_data.columns:
        if col in ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']:
            feature_values[col] = st.number_input(f'{col} for Analysis', min_value=float(df[col].min()), max_value=float(df[col].max()), value=float(input_data[col].iloc[0]))
        else:
            feature_values[col] = st.selectbox(f'{col} for Analysis', df[col].unique(), index=df[col].unique().tolist().index(input_data[col].iloc[0]))
    
    if st.button('Analyze Probability 📊'):
        analysis_data = pd.DataFrame([feature_values])
        analysis_encoded = analysis_data.copy()
        for col in categorical_columns:
            analysis_encoded[col] = le.fit_transform(analysis_encoded[col])
        prob = lr_model.predict_proba(analysis_encoded)[0][1]
        st.write(f'**Predicted Probability**: {prob:.2%}')
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#E74C3C" if prob >= 0.5 else "#2ECC71"}}
        ))
        fig.update_layout(title='Probability Gauge')
        st.plotly_chart(fig)

    st.markdown('</div>', unsafe_allow_html=True)

# Data Overview
elif page == 'Data Overview 📊':
    st.header('Data Overview 📊')
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('Dataset Preview')
    st.dataframe(df.head())
    
    st.subheader('Filter by Age Range')
    age_min, age_max = st.slider('Select Age Range', int(df['Age'].min()), int(df['Age'].max()), (int(df['Age'].min()), int(df['Age'].max())))
    filtered_df = df[(df['Age'] >= age_min) & (df['Age'] <= age_max)]
    st.write(f'**Filtered Records**: {len(filtered_df)}')
    
    st.subheader('Visualizations')
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(filtered_df, names='HeartDisease', title='Heart Disease Distribution')
        st.plotly_chart(fig)
    with col2:
        fig = px.histogram(filtered_df, x='Age', color='HeartDisease', title='Age Distribution')
        st.plotly_chart(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# Scenario Manager
elif page == 'Scenario Manager ✅':
    st.header('Scenario Manager ✅')
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if st.session_state.scenarios:
        scenario_names = [s['name'] for s in st.session_state.scenarios]
        selected_scenario = st.selectbox('Select Scenario', scenario_names)
        scenario = next(s for s in st.session_state.scenarios if s['name'] == selected_scenario)
        st.write('**Scenario Details**:')
        st.json(scenario['data'])
        st.write(f'**Random Forest Prediction**: {"Heart Disease" if scenario["rf_pred"] == 1 else "No Heart Disease"}')
        st.write(f'**Logistic Regression Prediction**: {"Heart Disease" if scenario["lr_pred"] == 1 else "No Heart Disease"}')
        st.write(f'**Logistic Regression Probability**: {scenario["lr_prob"]:.2%}')
        
        if st.button('Delete Scenario 🗑️'):
            st.session_state.scenarios = [s for s in st.session_state.scenarios if s['name'] != selected_scenario]
            st.success(f'Scenario "{selected_scenario}" deleted!')
    else:
        st.write('No scenarios saved yet.')
    st.markdown('</div>', unsafe_allow_html=True)

# Model Performance
elif page == 'Model Performance 📈':
    st.header('Model Performance 📈')
    st.markdown('<div class="card">', unsafe_allow_html=True)
    rf_pred = rf_model.predict(X_test)
    lr_pred = lr_model.predict(X_test)
    
    st.subheader('Random Forest Performance')
    st.write(f'**Accuracy**: {accuracy_score(y_test, rf_pred):.2%}')
    st.write('**Classification Report**:')
    st.text(classification_report(y_test, rf_pred))
    
    st.subheader('Logistic Regression Performance')
    st.write(f'**Accuracy**: {accuracy_score(y_test, lr_pred):.2%}')
    st.write('**Classification Report**:')
    st.text(classification_report(y_test, lr_pred))
    
    st.subheader('Feature Importance (Random Forest)')
    importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})
    fig = px.bar(importance, x='Feature', y='Importance', title='Feature Importance')
    st.plotly_chart(fig)
    
    st.subheader('ROC Curve')
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:,1])
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_model.predict_proba(X_test)[:,1])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rf_fpr, y=rf_tpr, name=f'Random Forest (AUC = {auc(rf_fpr, rf_tpr):.2f})'))
    fig.add_trace(go.Scatter(x=lr_fpr, y=lr_tpr, name=f'Logistic Regression (AUC = {auc(lr_fpr, lr_tpr):.2f})'))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name='Random', line=dict(dash='dash')))
    fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    st.plotly_chart(fig)
    st.markdown('</div>', unsafe_allow_html=True)