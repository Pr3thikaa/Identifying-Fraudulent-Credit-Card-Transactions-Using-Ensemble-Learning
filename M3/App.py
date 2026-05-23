import streamlit as st
import sqlite3
import bcrypt
import pandas as pd
import tensorflow as tf
import joblib
from catboost import CatBoostClassifier


st.set_page_config(page_title="Streamlit Fraud Detection", layout="centered")


def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode(), hashed_password.encode())

def add_user(name, username, password):
    with sqlite3.connect('users.db', check_same_thread=False) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (name TEXT, username TEXT, password TEXT)''')
        c.execute('SELECT * FROM users WHERE username = ?', (username,))
        if c.fetchone():
            st.error("Username already exists. Please choose a different username.")
            return
        hashed_password = hash_password(password)
        c.execute('INSERT INTO users (name, username, password) VALUES (?, ?, ?)', (name, username, hashed_password))
        conn.commit()
        st.success("Account created successfully! Please login...")

def login_user(username, password):
    with sqlite3.connect('users.db', check_same_thread=False) as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        if user and verify_password(password, user[2]):
            return user
    return None

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'page' not in st.session_state:
    st.session_state.page = "Login"

@st.cache(allow_output_mutation=True)
def load_models():
    catboost_model = CatBoostClassifier()
    catboost_model.load_model("./Model/Machine_Learning/catboost_model.cbm")

    standard_scaler = joblib.load("./Scalers/standard_scaler.joblib")
    min_max_scaler = joblib.load("./Scalers/minmax_scaler.joblib")

    cnn_model = tf.keras.models.load_model('./Model/Deep_Learning/cnn_model_99.keras', compile=False)

    return catboost_model, standard_scaler, min_max_scaler, cnn_model

catboost_model, standard_scaler, min_max_scaler, cnn_model = load_models()

def predict_fraud(sample_data):
    X_sample_scaled = standard_scaler.transform(sample_data)
    X_sample_leaf = catboost_model.calc_leaf_indexes(X_sample_scaled)
    sample_data_scaled = min_max_scaler.transform(X_sample_leaf)
    y_pred_prob = cnn_model.predict(sample_data_scaled)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    if int(y_pred[0][0]) == 1:
        return "Fraud", "🚨 **Fraud Transaction Detected!**", "⚠️ This transaction is fraudulent. Please check it."
    else:
        return "Non-Fraud", "✅ **Transaction is Safe!**", "😊 The transaction appears legitimate."

if st.session_state.authenticated:
    st.session_state.page = st.sidebar.selectbox(
        "Navigation", ["Fraud Detection", "Logout"], index=0)
else:
    st.session_state.page = st.sidebar.selectbox(
        "Navigation", ["Login", "Register"], index=0)

if st.session_state.page == "Login" and not st.session_state.authenticated:
    st.title("🔐 Streamlit Authentication")
    st.subheader("Login to your account")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")
    
    if login_button:
        user = login_user(username, password)
        if user:
            st.session_state.authenticated = True
            st.session_state.username = user[1]
            st.success(f"Welcome {user[0]}! Redirecting to Fraud Detection page...")
            st.session_state.page = "Fraud Detection"
            st.experimental_rerun()
        else:
            st.error("Invalid username or password.")


import time

if st.session_state.page == "Register" and not st.session_state.authenticated:
    st.title("🔐 Create a New Account")
    
    name = st.text_input("Full Name")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    register_button = st.button("Register")
    
    if register_button:
        if password != confirm_password:
            st.error("Passwords do not match!")
        elif len(password) < 6:
            st.error("Password must be at least 6 characters long.")
        else:
            add_user(name, username, password)
            
            

if st.session_state.page == "Fraud Detection" and st.session_state.authenticated:
    st.sidebar.success(f"Logged in as {st.session_state.username}")
    
    st.subheader("🚦 Fraud Detection Page")
    uploaded_file = st.file_uploader("📂 Upload a CSV file for prediction", type=["csv"])

    if uploaded_file:
        sample_data = pd.read_csv(uploaded_file)
        st.write("📝 Uploaded Data Preview:")
        st.dataframe(sample_data.head())
        
        if st.button("Predict Fraud"):
            prediction, status, message = predict_fraud(sample_data)
            
            if prediction == "Fraud":
                st.error(status)
                st.markdown(f"### {message}")
            else:
                st.success(status)
                st.markdown(f"### {message}")


if st.session_state.page == "Logout" and st.session_state.authenticated:
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.page = "Login"
    st.success("You have been logged out successfully!")
    st.experimental_rerun()

if not st.session_state.authenticated and st.session_state.page == "Fraud Detection":
    st.warning("⚠️ **Please log in with your account to access the Fraud Detection page.**")
    st.session_state.page = "Login"
    st.experimental_rerun()
