import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import LambdaCallback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os
from datetime import datetime, timedelta
import shap

@st.cache_data
def load_data():
    try:
        data = pd.read_csv('updated_merged_table_data.csv', encoding='latin1', on_bad_lines='skip')
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

data = load_data()

st.title("📊 Publishing Industry Analytics")
st.write("### Dataset Overview")

if not data.empty:
    st.dataframe(data.head())
else:
    st.error("No data loaded. Please check the dataset file.")

def preprocess_data(df):
    df = df.copy()
    df.fillna(method='ffill', inplace=True)

    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        df['order_year'] = df['order_date'].dt.year
        df['order_month'] = df['order_date'].dt.month
        df['order_day'] = df['order_date'].dt.day
        df.drop(columns=['order_date'], inplace=True)

    if 'order_timestamp' in df.columns:
        df.drop(columns=['order_timestamp'], inplace=True)

    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            try:
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column].astype(str))
                label_encoders[column] = le
            except Exception as e:
                st.error(f"Error encoding {column}: {e}")

    return df, label_encoders

def create_customer_churn(df):
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    churn_threshold_date = datetime.now() - timedelta(days=90)
    churned_customers = df.groupby('customer_id')['order_date'].max()
    churned_customers = churned_customers[churned_customers < churn_threshold_date]
    df['customer_churn'] = df['customer_id'].apply(lambda x: 1 if x in churned_customers.index else 0)
    return df

selected_columns = ['order_line_price', 'city', 'country_name', 'order_year', 'order_month', 'order_day']

if not data.empty and all(col in data.columns for col in ['order_line_price', 'city', 'country_name', 'customer_id']):
    data = create_customer_churn(data)
    st.write("### Churn Label Distribution")
    st.write(data['customer_churn'].value_counts())
    data, label_encoders = preprocess_data(data)
else:
    st.error("Missing required columns in dataset.")

def visualize_churn_identification(df):
    churn_counts = df['customer_churn'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(['Active Customers', 'Churned Customers'], churn_counts, color=['green', 'red'])
    ax.set_title('Churn Identification: Active vs. Churned Customers', fontsize=14, weight='bold')
    st.pyplot(fig)

def build_ann(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if not data.empty and st.button("🚀 Train Churn Model"):
    if 'customer_churn' in data.columns:
        X = data[selected_columns]
        y = data['customer_churn']

        try:
            X_encoded, _ = preprocess_data(X)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_encoded)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            model = build_ann(X_train.shape[1])

            epochs = st.slider("Select Number of Epochs", 5, 50, 10)
            progress_bar = st.progress(0)

            progress_callback = LambdaCallback(
                on_epoch_end=lambda epoch, logs: progress_bar.progress((epoch + 1) / epochs)
            )

            class_weights = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weights_dict = dict(enumerate(class_weights))

            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=32,
                validation_split=0.1,
                class_weight=class_weights_dict,
                callbacks=[progress_callback]
            )

            model.save('churn_model.h5')
            with open('scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            np.save('background_sample.npy', X_train[:100])  # Save background for SHAP

            st.success("✅ Churn Prediction Model Trained and Saved!")

            st.write("### Training and Validation Loss Over Epochs")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
            ax.plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
            ax.set_title('Loss vs. Epochs', fontsize=16, weight='bold')
            ax.legend()
            st.pyplot(fig)

            y_pred = (model.predict(X_test) > 0.5).astype("int32")
            st.write("### Model Evaluation Metrics:")
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
            st.write(f"Precision: {precision_score(y_test, y_pred):.2f}")
            st.write(f"Recall: {recall_score(y_test, y_pred):.2f}")
            st.write(f"F1 Score: {f1_score(y_test, y_pred):.2f}")

        except Exception as e:
            st.error(f"Error during model training: {e}")
    else:
        st.error("The target column 'customer_churn' is missing.")

enable_predictions = st.checkbox("🔍 Enable Churn Prediction Interface")
if enable_predictions and not data.empty:
    st.write("### Predict Customer Churn")
    visualize_churn_identification(data)

    input_df = pd.DataFrame({
        'order_line_price': [st.number_input("Order Line Price", min_value=0.0, step=0.01)],
        'city': [st.selectbox("City", label_encoders['city'].classes_)],
        'country_name': [st.selectbox("Country", label_encoders['country_name'].classes_)],
        'order_year': [st.number_input("Order Year", min_value=2000, max_value=2100, value=2024)],
        'order_month': [st.number_input("Order Month", min_value=1, max_value=12, value=4)],
        'order_day': [st.number_input("Order Day", min_value=1, max_value=31, value=1)]
    })

    if st.button("🔮 Predict Churn"):
        if not os.path.exists('churn_model.h5') or not os.path.exists('scaler.pkl') or not os.path.exists('background_sample.npy'):
            st.warning("Please train the model first before making predictions.")
        else:
            model = load_model('churn_model.h5')
            scaler = pickle.load(open('scaler.pkl', 'rb'))

            try:
                input_processed = input_df.copy()
                for col in ['city', 'country_name']:
                    le = label_encoders[col]
                    input_processed[col] = le.transform(input_processed[col])

                input_processed = input_processed[selected_columns]
                input_scaled = scaler.transform(input_processed)

                st.write("Scaled Input Data:", input_scaled)

                prediction = model.predict(input_scaled)[0][0]
                st.write(f"🧠 Predicted Churn Probability: `{prediction:.2f}`")

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(['Churn Probability'], [prediction], color=['blue'])
                ax.set_ylim(0, 1)
                ax.set_ylabel("Probability")
                st.pyplot(fig)

                st.subheader("🧩 SHAP Explanation (KernelExplainer)")

                background = np.load('background_sample.npy')
                explainer = shap.KernelExplainer(lambda x: model.predict(x).flatten(), background)
                shap_values = explainer.shap_values(input_scaled)

                fig, ax = plt.subplots()
                shap.summary_plot(shap_values, input_scaled, feature_names=input_processed.columns, show=False)
                st.pyplot(fig)  # ✅ FIXED here to avoid Streamlit warning

            except Exception as e:
                st.error(f"Error during prediction: {e}")
