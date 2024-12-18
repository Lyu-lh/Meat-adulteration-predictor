import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
import warnings
# warnings.filterwarnings('ignore')
# streamlit run C:\Users\lyu_l\Desktop\python\pythonProject1\app\app.py [ARGUMENTS]
h=""
# h=""
# Load the fetal state model
# model = joblib.load(r'./BP-ANN5.pkl')
model = joblib.load(h+'BP-ANN5.pkl')
# Define feature names
feature_names = [
    "L*", "C*", "S12", "Thr", "Lys", "His"]

data = pd.read_csv(h+"data10.csv").iloc[:, :-1].values
# Streamlit user interface
st.title("Meat adulteration predictor")
# Input features
Adhesiveness = st.number_input("L*:", format="%.5f", value=0.00000)
Aw = st.number_input("C*:", format="%.5f",value=0.00000)
a_value = st.number_input("S12:", format="%.5f",value=0.00000)
b_value = st.number_input("Thr:", format="%.5f",value=0.00000)
S9 = st.number_input("Lys:", format="%.5f",value=0.00000)
His = st.number_input("His:", format="%.5f",value=0.00000)

# Collect input values into a list
feature_values = [Adhesiveness, Aw, a_value, b_value, S9, His]

# Convert feature values into a DataFrame
features_df = pd.DataFrame([feature_values], columns=feature_names)

if st.button("Predict"):
    # Predict class and probabilities using DataFrame
    predicted_class = np.argmax(model.predict(features_df)[0])
    predicted = model.predict(features_df)[0]
    l = ["0%", "10%", "15%","20%","30%","40%", "50%", "70%","90%", "100%"]
    # Display prediction results
    st.write(f"**Predicted Class:** {l[predicted_class]}")
    st.write(f"**Prediction Probabilities:** {predicted}",format="%.5f")
    print(predicted_class)
    # Generate advice based on prediction results
    probability = predicted[predicted_class] * 100

    if predicted_class == 9:
        advice = (
            f"According to our model, the Patty is in an adulterated state. "
            f"The model predicts that the Patty has a {probability:.4f}% probability of being adulterated. "
        )
    elif predicted_class == 0:
        advice = (
            f"According to our model, the Patty is in a normal state. "
            f"The model predicts that the Patty has a {probability:.4f}% probability of being normal. "
        )
    else:
        advice = (
            f"According to our model, the Patty is in a suspicious state. "
            f"The model predicts that the Patty has a {probability:.4f}% probability of being suspicious. "
        )
    st.write(advice)
    explainer = shap.KernelExplainer(model.predict, data[:50])  # 使用一部分数据作为背景数据
    # 计算 SHAP 值
    shap_values = explainer.shap_values(features_df.values)
    print(shap_values)
    shap_value = shap_values[0, :, predicted_class]  # 提取该类别的 SHAP 值
    base_value = explainer.expected_value[0]  # 基线值
    # 绘制瀑布图
    shap.waterfall_plot(shap.Explanation(values=shap_value,
                                         base_values=base_value,
                                         data=features_df.values[0],
                                         feature_names=feature_names))
    plt.savefig("shap_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_plot.png")
