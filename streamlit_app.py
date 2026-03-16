import streamlit as st
import pandas as pd
import pickle

# --- PAGE CONFIG ---
st.set_page_config(page_title="California Housing Predictor", layout="wide")

# --- 1. LOAD THE MODEL ---
@st.cache_resource # This keeps the model in memory so it doesn't reload every time
def load_model():
    # Get the directory where this script is located
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, 'california_knn_pipeline.pkl')
    
    # Open the file safely
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# --- 2. UI HEADER ---
st.title("🏠 California Housing Price Predictor")
st.markdown("""
This app uses a **K-Nearest Neighbors (KNN)** model to predict the median house value in California based on 1990 Census data.
""")

st.divider()

# --- 3. USER INPUTS ---
st.sidebar.header("🔧 Input Housing Features")

def user_input_features():
    med_inc = st.sidebar.slider("Median Income (in $10,000s)", 0.5, 15.0, 3.5)
    house_age = st.sidebar.slider("Median House Age", 1, 52, 28)
    ave_rooms = st.sidebar.slider("Average Rooms", 1.0, 10.0, 5.0)
    ave_bedrms = st.sidebar.slider("Average Bedrooms", 1.0, 5.0, 1.0)
    population = st.sidebar.number_input("Population", 3, 35000, 1400)
    ave_occup = st.sidebar.slider("Average Occupancy", 1.0, 10.0, 3.0)
    latitude = st.sidebar.number_input("Latitude", 32.5, 42.0, 34.0)
    longitude = st.sidebar.number_input("Longitude", -124.3, -114.3, -118.0)

    data = {
        'MedInc': med_inc,
        'HouseAge': house_age,
        'AveRooms': ave_rooms,
        'AveBedrms': ave_bedrms,
        'Population': population,
        'AveOccup': ave_occup,
        'Latitude': latitude,
        'Longitude': longitude
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- 4. DISPLAY INPUTS & PREDICTION ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Selected Input Data")
    st.write(input_df)

with col2:
    st.subheader("Prediction Result")
    if st.button("💡 Predict House Value"):
        prediction = model.predict(input_df)
        
        # Format the output (Model outputs in $100,000s)
        final_price = prediction[0] * 100000
        
        st.metric(label="Estimated Median House Value", value=f"${final_price:,.2f}")
        st.success(f"The predicted value for this area is approximately ${final_price:,.2f}")

# --- 5. DATA INSIGHTS (Optional) ---
st.divider()
st.subheader("📍 Location context")
st.map(input_df, latitude="Latitude", longitude="Longitude")# Shows a map of the selected Latitude/Longitude