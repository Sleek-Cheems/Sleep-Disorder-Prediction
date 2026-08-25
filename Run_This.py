import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==========================================
# 1. SETUP & LOAD MODULES
# ==========================================
st.set_page_config(page_title="Sleep Health Pro", page_icon="🛌", layout="wide")

# Load the trained brains
try:
    model = joblib.load('knn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    target_encoder = joblib.load('target_encoder.pkl')
except FileNotFoundError:
    st.error("❌ Error: .pkl files not found. Please run the training script first!")
    st.stop()

st.title("🛌 Advanced Sleep Disorder Prediction System")
st.markdown("Enter the patient's lifestyle and medical details below to predict **Insomnia** or **Sleep Apnea**.")

# ==========================================
# 2. USER INPUTS (Organized by Category)
# ==========================================

# We use columns and expanders to make the 30 inputs look clean
col1, col2 = st.columns(2)

with col1:
    with st.expander("👤 1. Demographics & Work", expanded=True):
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 90, 30)
        occupation = st.selectbox("Occupation", label_encoders['Occupation'].classes_)
        work_hours = st.slider("Work Hours/Week", 10, 80, 40)
        commute_time = st.slider("Commute Time (mins)", 0, 120, 30)

    with st.expander("🍔 2. Lifestyle Habits"):
        smoker = st.selectbox("Smoker", ["Yes", "No"])
        alcohol = st.slider("Alcohol (Units/Week)", 0, 20, 2)
        caffeine = st.number_input("Caffeine Intake (mg)", 0, 500, 50)
        breakfast = st.selectbox("Breakfast Habit", ["Daily", "Sometimes", "Never"])
        activity_level = st.slider("Physical Activity Level (1-100)", 0, 100, 50)
        daily_steps = st.number_input("Daily Steps", 0, 20000, 5000)

with col2:
    with st.expander("❤️ 3. Medical Vitals", expanded=True):
        bmi_cat = st.selectbox("BMI Category", label_encoders['BMI Category'].classes_)
        heart_rate = st.slider("Heart Rate (bpm)", 40, 120, 70)
        systolic = st.number_input("Systolic BP (Upper)", 90, 180, 120)
        diastolic = st.number_input("Diastolic BP (Lower)", 60, 120, 80)
        oxygen = st.slider("Oxygen Saturation (%)", 80, 100, 98)
        neck_circ = st.number_input("Neck Circumference (cm)", 30.0, 60.0, 40.0)
        meds = st.selectbox("Medication", label_encoders['Medication_Usage'].classes_)
        family_hist = st.selectbox("Family History of Sleep Issues?", ["Yes", "No"])

    with st.expander("🌙 4. Sleep Metrics & Environment"):
        sleep_dur = st.slider("Sleep Duration (Hours)", 3.0, 10.0, 7.0)
        sleep_qual = st.slider("Quality of Sleep (1-10)", 1, 10, 6)
        stress = st.slider("Stress Level (1-10)", 1, 10, 5)
        nap_dur = st.slider("Nap Duration (mins)", 0, 120, 0)
        snoring = st.selectbox("Snoring Frequency", ["Never", "Rarely", "Sometimes", "Often"])
        screen_time = st.slider("Screen Time Before Bed (mins)", 0, 180, 30)
        room_temp = st.slider("Room Temp (°C)", 15.0, 30.0, 21.0)
        mattress_age = st.slider("Mattress Age (Years)", 0, 15, 5)
        noise_db = st.slider("Bedroom Noise (dB)", 20, 80, 40)
        hydration = st.number_input("Daily Hydration (Liters)", 0.5, 5.0, 2.0)

# ==========================================
# 3. PREDICTION LOGIC
# ==========================================

if st.button("🔮 Analyze Sleep Health"):
    # 1. Gather all inputs into a Dictionary
    # IMPORTANT: The keys (Left Side) must match the DataFrame columns from training EXACTLY
    input_data = {
        'Gender': gender,
        'Age': age,
        'Occupation': occupation,
        'Sleep Duration': sleep_dur,
        'Quality of Sleep': sleep_qual,
        'Physical Activity Level': activity_level,
        'Stress Level': stress,
        'BMI Category': bmi_cat,
        'Heart Rate': heart_rate,
        'Daily Steps': daily_steps,
        'Caffeine_Intake_mg': caffeine,
        'Alcohol_Consumption_Weekly': alcohol,
        'Smoker': smoker,
        'Screen_Time_Before_Bed_mins': screen_time,
        'Room_Temp_Celsius': room_temp,
        'Mattress_Age_Years': mattress_age,
        'Bedroom_Noise_dB': noise_db,
        'Daily_Hydration_L': hydration,
        'Medication_Usage': meds,
        'Nap_Duration_mins': nap_dur,
        'Work_Hours_Weekly': work_hours,
        'Commute_Time_Daily_mins': commute_time,
        'Breakfast_Habit': breakfast,
        'Family_History': family_hist,
        'Snoring_Frequency': snoring,
        'Oxygen_Saturation': oxygen,
        'Neck_Circumference_cm': neck_circ,
        'Systolic': systolic,
        'Diastolic': diastolic
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # --- 🛡️ PROFESSIONAL ERROR HANDLING START ---

    # 2. Encode Categorical Variables
    try:
        for col, le in label_encoders.items():
            if col in input_df.columns:
                # Check if the user input exists in the encoder's known classes
                val = input_df[col].iloc[0]
                if val not in le.classes_:
                    # 🛑 STOP EXECUTION if unseen category found (Safety Net)
                    st.error(
                        f"❌ Error: The value '{val}' for '{col}' was not seen during training. The model cannot process it.")
                    st.stop()
                else:
                    input_df[col] = le.transform(input_df[col])
    except Exception as e:
        st.error(f"Encoding Error: {e}")
        st.stop()

    # 3. Scale Features
    try:
        # Reorder columns to match training data exactly
        if hasattr(scaler, 'feature_names_in_'):
            input_df = input_df[scaler.feature_names_in_]

        # Transform using the scaler loaded from the backend
        scaled_data = scaler.transform(input_df)

    except Exception as e:
        st.error(f"❌ Scaling Error: Input features do not match model expectations.\nDetails: {e}")
        st.stop()

    # --- 🛡️ PROFESSIONAL ERROR HANDLING END ---

    # 4. Predict
    try:
        prediction_encoded = model.predict(scaled_data)
        prediction_label = target_encoder.inverse_transform(prediction_encoded)[0]

        # 5. Display Results & Dynamic Recommendations
        st.markdown("---")

        # --- LOGIC: GENERATE TAILORED ADVICE ---
        advice_list = []

        # Check Lifestyle Triggers
        if caffeine > 200:
            advice_list.append(
                "☕ **Caffeine Intake:** Your intake is high (>200mg). Avoid caffeine 6 hours before bed.")
        if screen_time > 60:
            advice_list.append(
                "📱 **Screen Time:** High exposure to blue light before bed disrupts melatonin. Try reading instead.")
        if bmi_cat in ["Overweight", "Obese"]:
            advice_list.append(
                "⚖️ **Weight Management:** Higher BMI is a top risk factor for Sleep Apnea. Weight reduction can significantly improve symptoms.")
        if smoker == "Yes":
            advice_list.append("🚭 **Smoking:** Smoking causes airway inflammation, worsening breathing during sleep.")
        if alcohol > 5:
            advice_list.append(
                "🍷 **Alcohol:** Alcohol reduces sleep quality and relaxes throat muscles, worsening Apnea.")
        if stress > 7:
            advice_list.append("🧠 **High Stress:** Stress is a major insomnia trigger. Consider mindfulness.")
        if noise_db > 50:
            advice_list.append("🔊 **Environment:** Your bedroom is too noisy (>50dB). Consider earplugs.")
        if systolic > 130 or diastolic > 85:
            advice_list.append(
                "❤️ **Blood Pressure:** Your BP is elevated. Sleep disorders and hypertension are linked.")

        # --- DISPLAY OUTPUTS ---

        if prediction_label == "None":
            st.success(f"## ✅ Prediction: Healthy (No Disorder Detected)")
            st.balloons()
            if advice_list:
                st.info("**Even though you are healthy, consider these improvements:**")
                for tip in advice_list:
                    st.write(tip)
            else:
                st.write("🌟 Great job! Your lifestyle and vitals indicate excellent sleep hygiene.")

        elif prediction_label == "Insomnia":
            st.warning(f"## ⚠️ Prediction: Insomnia Detected")
            st.markdown("### 🩺 Recommended Actions:")
            st.write("1. **Establish a Routine:** Go to bed and wake up at the same time every day.")
            st.write("2. **Cognitive Behavioral Therapy (CBT-I):** The gold standard treatment for insomnia.")

            if advice_list:
                st.markdown("### 🛠️ Personal Triggers Identified:")
                for tip in advice_list:
                    st.write(tip)

        else:  # Sleep Apnea
            st.error(f"## 🚨 Prediction: Sleep Apnea Detected")
            st.markdown("### 🩺 Immediate Medical Actions:")
            st.write("1. **Consult a Sleep Specialist:** You may need a Polysomnography (Sleep Study).")
            st.write("2. **Check Oxygen Levels:** Low oxygen during sleep puts strain on your heart.")

            if advice_list:
                st.markdown("### 🛠️ Risk Factors Identified:")
                for tip in advice_list:
                    st.write(tip)

    except Exception as e:
        st.error(f"Prediction Error: {e}")