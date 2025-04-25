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

# Custom CSS for professional medical analysis theme with smooth transitions
st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');

/* Main app styling */
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

/* Sidebar styling */
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

/* Headers */
h1, h2, h3 {
    color: #4A90E2;
    display: flex;
    align-items: center;
}
h1::before, h2::before, h3::before {
    content: url('https://img.icons8.com/color/36/000000/heart-monitor.png');
    margin-right: 10px;
}

/* Cards */
.card {
    background: #FFFFFF;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
    border: 1px solid #E0E0E0;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
    animation: cardFadeIn 0.5s ease-in;
}
.card:hover {
    border-color: #2ECC71;
    box-shadow: 0 4px 15px rgba(46, 204, 113, 0.2);
    transform: translateY(-3px);
}
@keyframes cardFadeIn {
    0% { opacity: 0; transform: translateY(10px); }
    100% { opacity: 1; transform: translateY(0); }
}

/* Buttons */
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

/* Inputs */
.stNumberInput > div > div > input, .stSelectbox > div > div > select, .stTextInput > div > div > input {
    background-color: #F9FAFB;
    color: #2C3E50;
    border: 1px solid #4A90E2;
    border-radius: 5px;
    transition: border-color 0.3s ease;
}
.stNumberInput:hover > div > div > input, .stSelectbox:hover > div > div > select, .stTextInput:hover > div > div > input {
    border-color: #2ECC71;
}

/* Text and tables */
.stMarkdown, .stDataFrame {
    color: #2C3E50;
}
.stDataFrame table {
    background-color: #FFFFFF;
    border: 1px solid #E0E0E0;
}

/* Success and warning messages */
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
    # Include the dataset in the repository to make it work on Streamlit Cloud
    df = pd.read_csv('heart.csv')
    # Handle zero cholesterol values
    df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)
    df['Cholesterol'].fillna(df['Cholesterol'].median(), inplace=True)
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    return df

# Train Random Forest and Logistic Regression models
@st.cache_resource
def train_model():
    df = load_data()
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    rf_report = classification_report(y_test, y_pred_rf, output_dict=True)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    rf_fpr, rf_tpr, _ = roc_curve(y_test, y_prob_rf)
    rf_roc_auc = auc(rf_fpr, rf_tpr)

    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, y_pred_lr)
    lr_report = classification_report(y_test, y_pred_lr, output_dict=True)
    y_prob_lr = lr_model.predict_proba(X_test)[:, 1]
    lr_fpr, lr_tpr, _ = roc_curve(y_test, y_prob_lr)
    lr_roc_auc = auc(lr_fpr, lr_tpr)

    return rf_model, rf_accuracy, rf_report, X.columns, rf_fpr, rf_tpr, rf_roc_auc, lr_model, lr_accuracy, lr_report, lr_fpr, lr_tpr, lr_roc_auc

# Initialize session state for scenarios
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = []

# Main app
def main():
    st.title('Heart Disease Prediction Dashboard 🩺')
    st.markdown('''
        <div style='text-align: center; margin-bottom: 20px;'>
            <img src='https://img.icons8.com/color/64/000000/ecg.png' style='vertical-align: middle;'>
            <p>A professional medical dashboard to analyze heart disease risk, perform what-if analysis, and compare scenarios.</p>
        </div>
    ''', unsafe_allow_html=True)

    # Load data and train models
    rf_model, rf_accuracy, rf_report, feature_cols, rf_fpr, rf_tpr, rf_roc_auc, lr_model, lr_accuracy, lr_report, lr_fpr, lr_tpr, lr_roc_auc = train_model()

    # Sidebar for navigation
    st.sidebar.header('Navigation 📋')
    section = st.sidebar.radio('Go to', ['Data Overview 📊', 'What-If Analysis 🩺', 'Scenario Manager ✅', 'Model Performance 📈'])

    if section == 'Data Overview 📊':
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header('Data Overview 📊')
        st.write('Summary statistics of the heart disease dataset:')
        st.dataframe(load_data().describe())

        st.subheader('Filter Data')
        age_range = st.slider('Select Age Range', int(load_data()['Age'].min()), int(load_data()['Age'].max()), (int(load_data()['Age'].min()), int(load_data()['Age'].max())), help='Filter patients by age range')
        filtered_df = load_data()[(load_data()['Age'] >= age_range[0]) & (load_data()['Age'] <= age_range[1])]
        st.write(f'Filtered dataset ({len(filtered_df)} rows):')
        st.dataframe(filtered_df.head())

        # Heart Disease Distribution
        fig1 = px.pie(load_data(), names='HeartDisease', title='Heart Disease Distribution (0 = No, 1 = Yes)',
                      template='plotly_white', color_discrete_sequence=['#4A90E2', '#E74C3C'])
        st.plotly_chart(fig1, use_container_width=True)

        # Age vs. Heart Disease
        fig2 = px.histogram(load_data(), x='Age', color='HeartDisease', barmode='overlay', title='Age Distribution by Heart Disease',
                            template='plotly_white', color_discrete_sequence=['#4A90E2', '#E74C3C'])
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    elif section == 'What-If Analysis 🩺':
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header('What-If Analysis 🩺')
        st.write('Adjust patient details to predict heart disease risk ❤️:')

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input('Age', min_value=20, max_value=100, value=40, help='Enter patient age')
            sex = st.selectbox('Sex', ['M', 'F'], help='Select patient sex')
            chest_pain = st.selectbox('Chest Pain Type', ['ATA', 'NAP', 'ASY', 'TA'], help='Select chest pain type')
            resting_bp = st.number_input('Resting Blood Pressure', min_value=0, value=120, help='Enter resting BP (mmHg)')
            cholesterol = st.number_input('Cholesterol', min_value=0, value=200, help='Enter cholesterol level (mg/dl)')
            fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1], help='Select fasting blood sugar status')
        with col2:
            resting_ecg = st.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'], help='Select ECG result')
            max_hr = st.number_input('Maximum Heart Rate', min_value=0, value=150, help='Enter max heart rate')
            exercise_angina = st.selectbox('Exercise-Induced Angina', ['N', 'Y'], help='Select angina status')
            oldpeak = st.number_input('Oldpeak', min_value=-2.0, max_value=10.0, value=0.0, step=0.1, help='Enter ST depression value')
            st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'], help='Select ST slope type')

        # Encode inputs
        input_data = {
            'Age': age,
            'Sex': 1 if sex == 'M' else 0,
            'ChestPainType': {'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3}[chest_pain],
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'RestingECG': {'Normal': 0, 'ST': 1, 'LVH': 2}[resting_ecg],
            'MaxHR': max_hr,
            'ExerciseAngina': 1 if exercise_angina == 'Y' else 0,
            'Oldpeak': oldpeak,
            'ST_Slope': {'Up': 0, 'Flat': 1, 'Down': 2}[st_slope]
        }

        input_df = pd.DataFrame([input_data])

        if st.button('Predict ❤️'):
            with st.spinner('Analyzing patient data...'):
                prediction = lr_model.predict(input_df)[0]
                probability = lr_model.predict_proba(input_df)[0][1]
                st.success(f'Prediction: {"Heart Disease" if prediction == 1 else "No Heart Disease"} ❤️')
                st.write(f'Probability of Heart Disease: {probability:.2%}')

        # Input-Based Probability (Logistic Regression) Graph
        st.subheader('Input-Based Probability (Logistic Regression) 📈')
        st.write('See the predicted probability of heart disease for the current patient details using Logistic Regression:')
        lr_probability = lr_model.predict_proba(input_df)[0][1]
        fig8 = go.Figure()
        fig8.add_trace(go.Scatter(x=[1], y=[lr_probability], mode='markers+text', name='Probability',
                                 marker={'color': '#F39C12', 'size': 15},
                                 text=[f'{lr_probability:.2%}'], textposition='top center'))
        fig8.update_layout(title='Heart Disease Probability for Current Inputs (Logistic Regression)',
                          xaxis_title='Input',
                          yaxis_title='Probability of Heart Disease',
                          template='plotly_white',
                          yaxis_tickformat='.2%',
                          xaxis={'showticklabels': False, 'range': [0.5, 1.5]},
                          yaxis={'range': [0, 1]})
        st.plotly_chart(fig8, use_container_width=True)

        # Save scenario
        scenario_name = st.text_input('Save this scenario (enter a name):', help='Enter a unique scenario name')
        if st.button('Save Scenario ✅') and scenario_name:
            st.session_state.scenarios.append({
                'name': scenario_name,
                **input_data,
                'probability': lr_model.predict_proba(input_df)[0][1],
                'prediction': 'Heart Disease' if lr_model.predict(input_df)[0] == 1 else 'No Heart Disease'
            })
            st.success(f'Scenario "{scenario_name}" saved! ✅')

        # Sensitivity Analysis (Random Forest)
        st.subheader('Sensitivity Analysis (Random Forest) 📈')
        st.write('See how changing one variable affects the predicted probability of heart disease using Random Forest:')
        variable_to_vary_rf = st.selectbox('Select variable to vary (Random Forest)', ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak'], help='Choose a variable to analyze its impact')
        ranges = {
            'Age': np.arange(20, 101, 5),
            'RestingBP': np.arange(0, 201, 10),
            'Cholesterol': np.arange(0, 401, 20),
            'MaxHR': np.arange(0, 201, 10),
            'Oldpeak': np.arange(-2.0, 10.1, 0.5)
        }
        values_rf = ranges[variable_to_vary_rf]
        probabilities_rf = []
        for value in values_rf:
            temp_data = input_data.copy()
            temp_data[variable_to_vary_rf] = value
            temp_df = pd.DataFrame([temp_data])
            prob = rf_model.predict_proba(temp_df)[0][1]
            probabilities_rf.append(prob)
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=values_rf, y=probabilities_rf, mode='lines+markers', name='Probability',
                                 line={'color': '#4A90E2'}, marker={'color': '#2ECC71'}))
        fig6.update_layout(title=f'Impact of {variable_to_vary_rf} on Heart Disease Probability (Random Forest)',
                          xaxis_title=variable_to_vary_rf,
                          yaxis_title='Probability of Heart Disease',
                          template='plotly_white',
                          yaxis_tickformat='.2%')
        st.plotly_chart(fig6, use_container_width=True)

        # Logistic Regression Probability Analysis
        st.subheader('Logistic Regression Probability Analysis 📈')
        st.write('Enter specific values to see how they affect the predicted probability of heart disease using the Logistic Regression model's fitted sigmoid function:')
        variable_to_vary_lr = st.selectbox('Select variable to vary (Logistic Regression)', ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak'], help='Choose a variable to analyze its impact')
        # Define valid ranges for validation
        valid_ranges = {
            'Age': (20, 100),
            'RestingBP': (0, 200),
            'Cholesterol': (0, 400),
            'MaxHR': (0, 200),
            'Oldpeak': (-2.0, 10.0)
        }
        min_val, max_val = valid_ranges[variable_to_vary_lr]
        # User inputs specific values
        values_input = st.text_input(f'Enter {variable_to_vary_lr} values (comma-separated, e.g., 40,50,60)', '40,50,60', help=f'Enter values between {min_val} and {max_val}')
        try:
            values_lr = [float(x.strip()) for x in values_input.split(',')]
            # Validate inputs
            if not values_lr:
                st.error('Please enter at least one value.')
            else:
                for val in values_lr:
                    if not (min_val <= val <= max_val):
                        st.error(f'Values must be between {min_val} and {max_val} for {variable_to_vary_lr}.')
                        break
                else:  # This else belongs to the for loop (executes if no break)
                    probabilities_lr = []
                    for value in values_lr:
                        temp_data = input_data.copy()
                        temp_data[variable_to_vary_lr] = value
                        temp_df = pd.DataFrame([temp_data])
                        prob = lr_model.predict_proba(temp_df)[0][1]
                        probabilities_lr.append(prob)
                    fig7 = go.Figure()
                    fig7.add_trace(go.Scatter(x=values_lr, y=probabilities_lr, mode='lines+markers', name='Probability',
                                             line={'color': '#E74C3C'}, marker={'color': '#4A90E2'}))
                    fig7.update_layout(title=f'Impact of {variable_to_vary_lr} on Heart Disease Probability (Logistic Regression)',
                                      xaxis_title=variable_to_vary_lr,
                                      yaxis_title='Probability of Heart Disease',
                                      template='plotly_white',
                                      yaxis_tickformat='.2%')
                    st.plotly_chart(fig7, use_container_width=True)
        except ValueError:
            st.error('Please enter valid numbers separated by commas (e.g., 40,50,60).')
        st.markdown('</div>', unsafe_allow_html=True)

    elif section == 'Scenario Manager ✅':
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header('Scenario Manager ✅')
        st.write('Compare patient scenarios to analyze heart disease predictions:')

        if not st.session_state.scenarios:
            st.warning('No scenarios saved yet. Use the What-If Analysis section to save scenarios. 🚨')
        else:
            # Display scenarios in a table
            scenario_df = pd.DataFrame(st.session_state.scenarios)
            st.write('Saved Scenarios:')
            st.dataframe(scenario_df.drop(columns=['name', 'prediction', 'probability']).style.format(precision=2))

            # Bar plot of probabilities
            fig3 = px.bar(scenario_df, x='name', y='probability', color='prediction',
                         title='Scenario Comparison: Heart Disease Probability',
                         labels={'probability': 'Probability', 'name': 'Scenario'},
                         text_auto='.2%', template='plotly_white',
                         color_discrete_sequence=['#4A90E2', '#E74C3C'])
            st.plotly_chart(fig3, use_container_width=True)

            # Delete scenario
            scenario_to_delete = st.selectbox('Delete a scenario:', [s['name'] for s in st.session_state.scenarios], help='Select a scenario to remove')
            if st.button('Delete Scenario 🗑️'):
                st.session_state.scenarios = [s for s in st.session_state.scenarios if s['name'] != scenario_to_delete]
                st.success(f'Scenario "{scenario_to_delete}" deleted! ✅')
        st.markdown('</div>', unsafe_allow_html=True)

    elif section == 'Model Performance 📈':
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header('Model Performance 📈')
        st.write(f'Random Forest Accuracy: {rf_accuracy:.2%}')
        st.write(f'Logistic Regression Accuracy: {lr_accuracy:.2%}')

        st.subheader('Random Forest Classification Report')
        st.json(rf_report)

        st.subheader('Logistic Regression Classification Report')
        st.json(lr_report)

        # Feature Importance (Random Forest)
        st.subheader('Feature Importance (Random Forest)')
        importance = pd.DataFrame({'Feature': feature_cols, 'Importance': rf_model.feature_importances_})
        fig4 = px.bar(importance, x='Importance', y='Feature', title='Feature Importance (Random Forest)', orientation='h',
                      template='plotly_white', color_discrete_sequence=['#2ECC71'])
        st.plotly_chart(fig4, use_container_width=True)

        # ROC Curve (Both Models)
        st.subheader('ROC Curve')
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=rf_fpr, y=rf_tpr, mode='lines', name=f'Random Forest (AUC = {rf_roc_auc:.2f})', line={'color': '#4A90E2'}))
        fig5.add_trace(go.Scatter(x=lr_fpr, y=lr_tpr, mode='lines', name=f'Logistic Regression (AUC = {lr_roc_auc:.2f})', line={'color': '#E74C3C'}))
        fig5.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line={'dash': 'dash', 'color': '#2C3E50'}))
        fig5.update_layout(title='Receiver Operating Characteristic (ROC) Curve',
                          xaxis_title='False Positive Rate',
                          yaxis_title='True Positive Rate',
                          template='plotly_white')
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
