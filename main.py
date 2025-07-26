import pandas as pd #pandas per gestire i dati
from sklearn.model_selection import train_test_split #per splittare i dati in training e test
from sklearn.ensemble import RandomForestClassifier #il modello
from sklearn.metrics import roc_auc_score #per calcolare la precisione del modello
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import streamlit as st


def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestClassifier(n_estimators = 100, random_state  = 0)
    model.fit(X_train, y_train)
    prediction = model.predict_proba(X_valid)[:,1]
    return roc_auc_score(y_valid, prediction), model

file_path = 'DataSet/healthcare-dataset-stroke-data.csv'

X = pd.read_csv(file_path)
y = X.stroke
X.drop(['stroke', 'id'], axis = 1, inplace = True)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 0)

# AGGIUNGO BMI MANCANTI
imputer = SimpleImputer(strategy = 'median')
X_train['bmi'] = imputer.fit_transform(X_train[['bmi']]) 
X_valid['bmi'] = imputer.transform(X_valid[['bmi']]) 


object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]



#
#  TOLGO TUTTE LE COLONNE PROBLEMATICHE
#

# tolgo le colonne dove manca un valore (bmi)
def remove_problematic():
    cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
    reduced_X_train = X_train.drop(cols_with_missing, axis = 1)
    reduced_X_valid = X_valid.drop(cols_with_missing, axis = 1)

    # tolgo le colonne che hanno valori non numerici
    drop_reduced_X_train = reduced_X_train.select_dtypes(exclude=['object'])
    drop_reduced_X_valid = reduced_X_valid.select_dtypes(exclude=['object'])

    prova_1 = (score_dataset(drop_reduced_X_train, drop_reduced_X_valid, y_train, y_valid))

#
#   APPLICO ORDINAL ENCODING
#
def ordinal_encoding():
    ordinal_encoder = OrdinalEncoder()

    X_ordenc_train = X_train.copy()
    X_ordenc_valid = X_valid.copy()

    X_ordenc_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
    X_ordenc_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])

    _, prova_2 = score_dataset(X_ordenc_train, X_ordenc_valid, y_train, y_valid)
    return prova_2, ordinal_encoder

#
#   APPLICO ONE HOT ENCODING
#

def one_hot_encoding():
    OH_encoder = OneHotEncoder(handle_unknown = 'ignore', sparse_output=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
    OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

    OH_cols_train.index = X_train.index
    OH_cols_valid.index = X_valid.index

    num_X_train = X_train.drop(object_cols, axis = 1)
    num_X_valid = X_valid.drop(object_cols, axis = 1)

    OH_X_train = pd.concat([num_X_train, OH_cols_train], axis = 1)
    OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis = 1)

    OH_X_train.columns = OH_X_train.columns.astype(str)
    OH_X_valid.columns = OH_X_valid.columns.astype(str)

    prova_3 = score_dataset(OH_X_train, OH_X_valid, y_train, y_valid)


#
#   APPLICO ONE HOT ENCODING E ORDINAL ENCODING
#
def OH_O_encoding():
    OH_cols = ['work_type', 'smoking_status']
    O_cols = [col for col in object_cols if col not in OH_cols]
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[OH_cols]))
    OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[OH_cols]))
    OH_cols_train.columns = OH_encoder.get_feature_names_out(OH_cols)
    OH_cols_valid.columns = OH_encoder.get_feature_names_out(OH_cols)
    OH_cols_train.index = X_train.index
    OH_cols_valid.index = X_valid.index


    ordinal_encoder_2 = OrdinalEncoder()
    X_ord_train = X_train.copy()
    X_ord_valid = X_valid.copy()
    X_ord_train[O_cols] = ordinal_encoder_2.fit_transform(X_train[O_cols])
    X_ord_valid[O_cols] = ordinal_encoder_2.transform(X_valid[O_cols])
    num_X_train = X_ord_train.drop(object_cols, axis = 1)
    num_X_valid = X_ord_valid.drop(object_cols, axis = 1)
    X_train_processed = pd.concat([num_X_train, X_ord_train[O_cols], OH_cols_train], axis = 1)
    X_valid_processed = pd.concat([num_X_valid, X_ord_valid[O_cols], OH_cols_valid], axis = 1)

    prova_4 = score_dataset(X_train_processed, X_valid_processed , y_train, y_valid)

final_model, ordinal_encoder = ordinal_encoding()

st.set_page_config('Stroke Predictor', layout = "centered")

if "show_content" not in st.session_state:
    st.session_state.show_content = False
if "lingua" not in st.session_state:
    st.session_state.lingua = None


if not st.session_state.show_content:
    st.title("Benvenuto! / Welcome!")
    st.write("Seleziona la lingua per iniziare / Choose the language to start")
    if st.button("English"):
        st.session_state.lingua = "inglese"
        st.session_state.show_content = True
        st.rerun()
    if st.button("Italiano"):
        st.session_state.show_content = True
        st.session_state.lingua = "italiano"
        st.rerun()
elif  st.session_state.lingua == "inglese":
    st.title("Stroke Predictor")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox('Sex', ['Male', 'Female', 'Other'])
        hypertension = st.selectbox('Hypertension', ['Yes', 'No'])
        age = st.slider('Age', 0, 100, 40)
        heart_disease = st.selectbox('Heart Disease', ['Yes', 'No'])
        ever_married = st.selectbox('Married', ['Yes', 'No'])
    with col2:
        work_type = st.selectbox('Type of work', ['Private', 'Self-employed', 'Govt job', 'children', 'Never worked'])
        residence_type = st.selectbox('Type of residence', ['Urban', 'Rural'])
        avg_glucose_level = st.number_input('Glucose Level', 50.0, 300.0, 100.0)
        bmi = st.number_input('BMI', 10.0, 60.0, 25.0)
        smoking_status = st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'Unkown'])

    user_data = pd.DataFrame({
        'gender': [gender],
        'age': [float(age)],
        'hypertension': [1 if hypertension == "Yes" else 0],
        'heart_disease': [1 if heart_disease == "Yes" else 0],
        'ever_married': [ever_married],
        'work_type': [work_type],
        'Residence_type': [residence_type],
        'avg_glucose_level': [float(avg_glucose_level)],
        'bmi': [float(bmi)],
        'smoking_status': [smoking_status]
    })


    user_data[object_cols] = ordinal_encoder.transform(user_data[object_cols])
    prob = final_model.predict_proba(user_data)[:,1][0]

    if st.button("Predict", help = "Predict stroke probability"):
        st.write("Probability of a stroke: ", prob*100, "%")


    st.write("DataSet:")
    st.write(pd.read_csv(file_path))
elif st.session_state.lingua == "italiano":
    st.title("Predittore di ictus")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox('Sesso', ['Maschio', 'Femmina', 'Altro'])
        hypertension = st.selectbox('Ipertensione?', ['Si', 'No'])
        age = st.slider('Età', 0, 100, 40)
        heart_disease = st.selectbox('Problemi di cuore?', ['Si', 'No'])
        ever_married = st.selectbox('Sei mai stato sposato?', ['Si', 'No'])
    with col2:
        work_type = st.selectbox('Tipo di lavoro', ['Privato', 'Autonomo', 'Dipendente statale', 'Bambino', 'Mai lavorato'])
        residence_type = st.selectbox('Tipo di residenza', ['Urbana', 'Rurale'])
        avg_glucose_level = st.number_input('Livello di glucosio nel sangue', 50.0, 300.0, 100.0)
        bmi = st.number_input("BMI (indice di massa corporea)", 10.0, 60.0, 25.0)
        smoking_status = st.selectbox('Stato di fumatore', ['Fumo', 'Mai fumato', 'Ho fumato ma ho smesso', 'Non lo so'])

    gender_map = {'Maschio': 'Male', 'Femmina': 'Female', 'Altro': 'Other'}
    work_type_map = {
        'Privato': 'Private',
        'Autonomo': 'Self-employed',
        'Dipendente statale': 'Govt job',
        'Bambino': 'children',
        'Mai lavorato': 'Never worked'
    }
    residence_type_map = {'Urbana': 'Urban', 'Rurale': 'Rural'}
    smoking_status_map = {
        'Fumo': 'smokes',
        'Mai fumato': 'never smoked',
        'Ho fumato ma ho smessod': 'formerly smoked',
        'Non lo so': 'Unknown' 
    }
    ever_married_map = {'Si': 'Yes', 'No': 'No'} # Assuming 'Yes'/'No' for married was original

    user_data = pd.DataFrame({
        'gender': [gender_map.get(gender, gender)], # Use .get() for safe mapping
        'age': [float(age)],
        'hypertension': [1 if hypertension == "Si" else 0],
        'heart_disease': [1 if heart_disease == "Si" else 0],
        'ever_married': [ever_married_map.get(ever_married, ever_married)],
        'work_type': [work_type_map.get(work_type, work_type)],
        'Residence_type': [residence_type_map.get(residence_type, residence_type)],
        'avg_glucose_level': [float(avg_glucose_level)],
        'bmi': [float(bmi)],
        'smoking_status': [smoking_status_map.get(smoking_status, smoking_status)]
    })


    user_data[object_cols] = ordinal_encoder.transform(user_data[object_cols])
    prob = final_model.predict_proba(user_data)[:,1][0]

    if st.button("Prevedi", help = "Prevede la probabilita di un ictus"):
        st.write("Probabilita di un ictus: ", prob*100, "%")


    st.write("Dati utilizzati per addestrare il modello:")
    st.write(pd.read_csv(file_path))
