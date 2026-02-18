import streamlit as st
import pickle

@st.cache_resource
def load_model():
    with open(r"C:\Users\aslam\OneDrive\Desktop\spam\spam_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()


st.set_page_config(
    page_title="Telegram Spam Detector",
    page_icon="📩",
    layout="centered"
)


st.title("📩 Telegram Spam / Ham Classifier")
st.write("Check whether a Telegram message is Spam or Ham")

st.divider()

# Text input
user_input = st.text_area("Enter your message here 👇", height=150)

# Predict button
if st.button("🔍 Check Message"):

    if user_input.strip() == "":
        st.warning("⚠ Please enter a message first.")
    else:
        prediction = model.predict([user_input])[0]

        st.divider()

        if prediction.lower() == "spam":
            st.error("🚨 This message is SPAM")
        else:
            st.success("✅ This message is HAM (Not Spam)")
