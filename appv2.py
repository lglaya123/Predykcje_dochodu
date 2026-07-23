import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# 1. Konfiguracja strony
st.set_page_config(
    page_title="Przewidywanie dochodu >50K",
    layout="wide"
)

# 2. Wczytanie wytrenowanego potoku ML (z buforowaniem st.cache_resource dla wydajności)
@st.cache_resource
def get_model():
    # PyCaret automatycznie szuka pliku z rozszerzeniem .pkl
    return load_model('income_pipeline')

pipeline = get_model()

# 3. Tytuł aplikacji
st.title("Aplikacja do prognozowania poziomu dochodu")
st.write("Wprowadź dane demograficzne i zawodowe, aby sprawdzić prawdopodobieństwo dochodu powyżej 50 000 USD rocznie.")

# 4. Formularz danych wejściowych
with st.form("income_form"):
    st.subheader("Dane wejściowe")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Wiek", min_value=17, max_value=90, value=35)
        workclass = st.selectbox(
            "Forma zatrudnienia (workclass)", 
            options=['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
        )
        education = st.selectbox(
            "Wykształcenie (education)",
            options=['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']
        )
        education_num = st.number_input("Liczba lat edukacji (education-num)", min_value=1, max_value=16, value=13)
        marital_status = st.selectbox(
            "Stan cywilny (marital-status)",
            options=['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
        )
        occupation = st.selectbox(
            "Zawód (occupation)",
            options=['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
        )
        relationship = st.selectbox(
            "Rola w rodzinie (relationship)",
            options=['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
        )

    with col2:
        race = st.selectbox(
            "Rasa (race)",
            options=['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
        )
        sex = st.selectbox(
            "Płeć (sex)",
            options=['Female', 'Male']
        )
        capital_gain = st.number_input("Zyski kapitałowe (capital-gain)", min_value=0, value=0)
        capital_loss = st.number_input("Straty kapitałowe (capital-loss)", min_value=0, value=0)
        hours_per_week = st.number_input("Liczba godzin pracy tygodniowo (hours-per-week)", min_value=1, max_value=99, value=40)
        native_country = st.selectbox(
            "Kraj pochodzenia (native-country)",
            options=['United-States', 'Cuba', 'Jamaica', 'India', 'Mexico', 'South', 'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran', 'Philippines', 'Italy', 'Poland', 'Columbia', 'Cambodia', 'Thailand', 'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal', 'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala', 'China', 'Japan', 'Yugoslavia', 'Peru', 'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago', 'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary', 'Holand-Netherlands']
        )

    submitted = st.form_submit_button("Przewiduj dochód")

# 5. Logika po kliknięciu przycisku
if submitted:
    # Tworzymy słownik z kluczami odpowiadającymi dokładnie nazwom kolumn w zbiorze treningowym
    input_data = {
        'age': age,
        'workclass': workclass,
        'education': education,
        'education-num': education_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': sex,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }
    
    # Konwersja na DataFrame (z jednym wierszem)
    input_df = pd.DataFrame([input_data])
    
    # Wykonanie predykcji za pomocą potoku PyCareta
    predictions = predict_model(estimator=pipeline, data=input_df)
    
    # Wyciągnięcie wyników
    # PyCaret tworzy kolumny 'prediction_label' oraz 'prediction_score'
    prediction_label = predictions['prediction_label'].iloc[0]
    prediction_score = predictions['prediction_score'].iloc[0]
    
    st.markdown("---")
    st.subheader("Wynik analizy:")
    
    # Prezentacja wyniku
    if str(prediction_label) in ['1', '>50K']:
        st.success(f"💰 **Prognoza: Dochód POWYŻEJ 50 000 USD rocznie (>50K)**")
    else:
        st.warning(f"📉 **Prognoza: Dochód PONIŻEJ lub RÓWNY 50 000 USD rocznie (<=50K)**")
        
    st.metric(label="Prawdopodobieństwo/Pewność predykcji", value=f"{prediction_score * 100:.2f}%")