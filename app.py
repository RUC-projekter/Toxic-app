import streamlit as st
import requests
import torch
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification

# Load your custom toxicity model
@st.cache_resource
def load_toxicity_model():
    model = BertForSequenceClassification.from_pretrained("moazla/bert-toxic-detection")  # Update with your Hugging Face model ID
    tokenizer = BertTokenizer.from_pretrained("moazla/bert-toxic-detection")  # Update with your Hugging Face model ID
    return model.to("cuda" if torch.cuda.is_available() else "cpu"), tokenizer

model, tokenizer = load_toxicity_model()

# Function to predict toxicity
def predict_toxicity(text):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    ).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs[0][1]  # Probability of being toxic

# Highlight toxic words using LIME
def get_lime_explanation(text):
    explainer = LimeTextExplainer(class_names=["Non-Toxic", "Toxic"])
    
    def predictor(texts):
        inputs = tokenizer(
            texts, return_tensors="pt", truncation=True, padding=True
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs).logits.cpu()
        return torch.softmax(outputs, dim=1).numpy()
    
    explanation = explainer.explain_instance(text, predictor, num_features=5)
    return explanation.as_list()

# Generate a bar chart for toxic words
def plot_lime_bar_chart(words, importances, title):
    fig, ax = plt.subplots()
    ax.barh(words, importances, color="red", alpha=0.7)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    st.pyplot(fig)

# Streamlit UI
st.title("Toxic Text Analyzer & Rephraser")
text = st.text_area("Enter text to analyze for toxicity and rephrase:", height=150)

if st.button("Analyze"):
    with st.spinner("Analyzing..."):
        # Predict toxicity
        prob_toxic = predict_toxicity(text)
        explanation = get_lime_explanation(text)
        st.session_state["original_text"] = text
        st.session_state["original_score"] = prob_toxic
        st.session_state["original_explanation"] = explanation

        # Display original results
        st.subheader(f"Original Text Toxicity Score: {prob_toxic:.2f}")
        toxic_words, importances = zip(*explanation)
        st.write("Most Toxic Words:", toxic_words)
        plot_lime_bar_chart(toxic_words, importances, "Toxic Words (Original)")

if st.session_state.get("original_text") and st.button("Rephrase Toxic Words"):
    with st.spinner("Rephrasing..."):
        try:
            # Call Render FastAPI for rephrasing
            response = requests.post(
                "https://toxic-app.onrender.com/rephrase",  # Replace with your actual Render API URL
                json={"text": st.session_state["original_text"]}
            )
            if response.status_code == 200:
                rephrased_text = response.json()["rephrased"]
                rephrased_score = predict_toxicity(rephrased_text)
                rephrased_explanation = get_lime_explanation(rephrased_text)

                # Display comparison
                st.subheader("Comparison Before and After Rephrasing")
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Original Text")
                    st.write(st.session_state["original_text"])
                    st.subheader(f"Toxicity Score: {st.session_state['original_score']:.2f}")
                    original_words, original_importances = zip(*st.session_state["original_explanation"])
                    plot_lime_bar_chart(original_words, original_importances, "Toxic Words (Original)")

                with col2:
                    st.subheader("Rephrased Text")
                    st.write(rephrased_text)
                    st.subheader(f"Toxicity Score: {rephrased_score:.2f}")
                    rephrased_words, rephrased_importances = zip(*rephrased_explanation)
                    plot_lime_bar_chart(rephrased_words, rephrased_importances, "Toxic Words (Rephrased)")

            else:
                st.error(f"Rephrasing failed: {response.text}")
        except Exception as e:
            st.error(f"Rephrasing error: {str(e)}")
