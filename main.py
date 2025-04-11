import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import base64
import json
import gtts

# Import disease summaries from separate file
from disease_summaries import disease_summaries

# Set page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# Available languages dictionary
languages = {
    "English": {
        "title": "🌿 Plant Disease Detection",
        "description": "Upload a leaf image to predict the disease.",
        "upload_prompt": "Choose an image...",
        "uploaded_caption": "Uploaded Image",
        "prediction_text": "🌱 Prediction:",
        "language_selector": "Select Language",
        "confidence": "Confidence",
        "no_model": "Model not found. Please check the model path.",
        "play_voice": "Play Voice Summary",
        "disease_summary": "Disease Summary"
    },
    "Hindi (हिन्दी)": {
        "title": "🌿 पौधे की बीमारी की पहचान",
        "description": "बीमारी का पता लगाने के लिए पत्ती की छवि अपलोड करें।",
        "upload_prompt": "एक छवि चुनें...",
        "uploaded_caption": "अपलोड की गई छवि",
        "prediction_text": "🌱 पूर्वानुमान:",
        "language_selector": "भाषा चुनें",
        "confidence": "विश्वास स्तर",
        "no_model": "मॉडल नहीं मिला। कृपया मॉडल पथ जांचें।",
        "play_voice": "वॉयस सारांश सुनें",
        "disease_summary": "रोग सारांश"
    },
    # Other languages preserved here...
    "Tamil (தமிழ்)": {
        "title": "🌿 செடியின் நோயறிதல்",
        "description": "நோயை கணிக்க ஒரு இலைப்படத்தை பதிவேற்றவும்.",
        "upload_prompt": "படத்தை தேர்வு செய்யவும்...",
        "uploaded_caption": "பதிவேற்றப்பட்ட படம்",
        "prediction_text": "🌱 கணிப்பு:",
        "language_selector": "மொழியைத் தேர்ந்தெடுக்கவும்",
        "confidence": "நம்பிக்கை",
        "no_model": "மாதிரி கிடைக்கவில்லை. பாதையைச் சரிபார்க்கவும்.",
        "play_voice": "குரல் சுருக்கம் இயக்கு",
        "disease_summary": "நோய் சுருக்கம்"
    },
       "Telugu (తెలుగు)": {
        "title": "🌿 మొక్కల వ్యాధి గుర్తింపు",
        "description": "రోగాన్ని అంచనా వేయడానికి ఆకుపై చిత్రాన్ని అప్‌లోడ్ చేయండి.",
        "upload_prompt": "చిత్రాన్ని ఎంచుకోండి...",
        "uploaded_caption": "అప్‌లోడ్ చేసిన చిత్రం",
        "prediction_text": "🌱 అంచనా:",
        "language_selector": "భాషను ఎంచుకోండి",
        "confidence": "నమ్మకం",
        "no_model": "మోడల్ కనుగొనబడలేదు. దయచేసి మోడల్ మార్గాన్ని తనిఖీ చేయండి.",
        "play_voice": "ధ్వని సారాంశాన్ని వినండి",
        "disease_summary": "వ్యాధి సారాంశం"
    },
    "Kannada (ಕನ್ನಡ)": {
        "title": "🌿 ಸಸ್ಯ ರೋಗ ಪತ್ತೆ",
        "description": "ರೋಗವನ್ನು ಊಹಿಸಲು ಎಲೆ ಚಿತ್ರದ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.",
        "upload_prompt": "ಚಿತ್ರವನ್ನು ಆಯ್ಕೆಮಾಡಿ...",
        "uploaded_caption": "ಅಪ್‌ಲೋಡ್ ಮಾಡಿದ ಚಿತ್ರ",
        "prediction_text": "🌱 ಊಹೆ:",
        "language_selector": "ಭಾಷೆ ಆಯ್ಕೆಮಾಡಿ",
        "confidence": "ಆತ್ಮವಿಶ್ವಾಸ",
        "no_model": "ಮಾದರಿ ಕಂಡುಬಂದಿಲ್ಲ. ದಯವಿಟ್ಟು ಮಾರ್ಗ ಪರಿಶೀಲಿಸಿ.",
        "play_voice": "ಧ್ವನಿ ಸಾರಾಂಶವನ್ನು ಆಡಿ",
        "disease_summary": "ರೋಗ ಸಾರಾಂಶ"
    },
    "Malayalam (മലയാളം)": {
        "title": "🌿 ചെടികളുടെ രോഗം തിരിച്ചറിയൽ",
        "description": "രോഗം കണക്കാക്കാൻ ഇലയുടെ ചിത്രം അപ്‌ലോഡ് ചെയ്യുക.",
        "upload_prompt": "ഒരു ചിത്രം തിരഞ്ഞെടുക്കുക...",
        "uploaded_caption": "അപ്‌ലോഡ് ചെയ്ത ചിത്രം",
        "prediction_text": "🌱 പ്രവചനം:",
        "language_selector": "ഭാഷ തിരഞ്ഞെടുക്കുക",
        "confidence": "വിശ്വാസം",
        "no_model": "മോഡൽ കണ്ടെത്താനായില്ല. ദയവായി പാത പരിശോധിക്കുക.",
        "play_voice": "ശബ്ദ സാരാംശം പ്ലേ ചെയ്യുക",
        "disease_summary": "രോഗത്തിന്റെ സാരാംശം"
    },
    "Bengali (বাংলা)": {
        "title": "🌿 উদ্ভিদের রোগ সনাক্তকরণ",
        "description": "রোগ নির্ণয় করতে একটি পাতার ছবি আপলোড করুন।",
        "upload_prompt": "একটি ছবি নির্বাচন করুন...",
        "uploaded_caption": "আপলোডকৃত ছবি",
        "prediction_text": "🌱 পূর্বাভাস:",
        "language_selector": "ভাষা নির্বাচন করুন",
        "confidence": "আস্থা",
        "no_model": "মডেল পাওয়া যায়নি। দয়া করে মডেল পথ পরীক্ষা করুন।",
        "play_voice": "ভয়েস সারাংশ চালান",
        "disease_summary": "রোগের সারাংশ"
    },
    "Marathi (मराठी)": {
        "title": "🌿 वनस्पती रोग ओळख",
        "description": "रोग ओळखण्यासाठी पानाचा फोटो अपलोड करा.",
        "upload_prompt": "फोटो निवडा...",
        "uploaded_caption": "अपलोड केलेला फोटो",
        "prediction_text": "🌱 अंदाज:",
        "language_selector": "भाषा निवडा",
        "confidence": "विश्वास",
        "no_model": "मॉडेल सापडले नाही. कृपया पथ तपासा.",
        "play_voice": "व्हॉइस सारांश प्ले करा",
        "disease_summary": "रोगाचा सारांश"
    },
    "Gujarati (ગુજરાતી)": {
        "title": "🌿 છોડની બીમારી ઓળખ",
        "description": "બીમારીનો અંદાજ લગાવવા માટે પાનની છબી અપલોડ કરો.",
        "upload_prompt": "છબી પસંદ કરો...",
        "uploaded_caption": "અપલોડ કરેલી છબી",
        "prediction_text": "🌱 અંદાજ:",
        "language_selector": "ભાષા પસંદ કરો",
        "confidence": "વિશ્વાસ",
        "no_model": "મોડેલ મળ્યું નથી. કૃપા કરીને પાથ તપાસો.",
        "play_voice": "આવાજ સારાંશ વગાડો",
        "disease_summary": "બીમારી સારાંશ"
    },
    "Punjabi (ਪੰਜਾਬੀ)": {
        "title": "🌿 ਪੌਦੇ ਦੀ ਬਿਮਾਰੀ ਦੀ ਪਹਿਚਾਣ",
        "description": "ਬਿਮਾਰੀ ਦਾ ਅਨੁਮਾਨ ਲਗਾਉਣ ਲਈ ਪੱਤੇ ਦੀ ਤਸਵੀਰ ਅਪਲੋਡ ਕਰੋ।",
        "upload_prompt": "ਇੱਕ ਤਸਵੀਰ ਚੁਣੋ...",
        "uploaded_caption": "ਅਪਲੋਡ ਕੀਤੀ ਤਸਵੀਰ",
        "prediction_text": "🌱 ਅਨੁਮਾਨ:",
        "language_selector": "ਭਾਸ਼ਾ ਚੁਣੋ",
        "confidence": "ਭਰੋਸਾ",
        "no_model": "ਮਾਡਲ ਨਹੀਂ ਮਿਲਿਆ। ਕਿਰਪਾ ਕਰਕੇ ਮਾਡਲ ਪਾਥ ਦੀ ਜਾਂਚ ਕਰੋ।",
        "play_voice": "ਵੌਇਸ ਸੰਖੇਪ ਚਲਾਓ",
        "disease_summary": "ਬਿਮਾਰੀ ਸੰਖੇਪ"
    },
    "Odia (ଓଡ଼ିଆ)": {
        "title": "🌿 ଉଦ୍ଭିଦ ରୋଗ ପରିଚୟ",
        "description": "ରୋଗ ପରିଚୟ ପାଇଁ ଗଛପତ୍ରର ଛବି ଅପଲୋଡ୍ କରନ୍ତୁ।",
        "upload_prompt": "ଏକ ଛବି ଚୟନ କରନ୍ତୁ...",
        "uploaded_caption": "ଅପଲୋଡ୍ ହୋଇଥିବା ଛବି",
        "prediction_text": "🌱 ଅନୁମାନ:",
        "language_selector": "ଭାଷା ବାଛନ୍ତୁ",
        "confidence": "ଭରସା",
        "no_model": "ମଡେଲ୍ ମିଳିଲା ନାହିଁ। ଦୟାକରି ପଥ ଯାଞ୍ଚ କରନ୍ତୁ।",
        "play_voice": "ଶବ୍ଦ ସାରାଂଶ ଚଲାନ୍ତୁ",
        "disease_summary": "ରୋଗ ସାରାଂଶ"
    }
}

def get_disease_summary(disease_name, language):
    try:
        # First try to get summary in selected language
        if language in disease_summaries:
            if disease_name in disease_summaries[language]:
                return disease_summaries[language][disease_name]
        
        # Fallback to English if not available in selected language
        if disease_name in disease_summaries["English"]:
            return disease_summaries["English"][disease_name]
            
        return "Detailed information about this disease is not available at the moment."
    except Exception as e:
        st.warning(f"Error loading disease summary: {e}")
        return "Detailed information about this disease is not available at the moment."

# Function to load model
@st.cache_resource
def load_model(model_path):
    try:
        if os.path.exists(model_path):
            return tf.keras.models.load_model(model_path)
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to preprocess image
def preprocess_image(image, target_size=(128, 128)):
    img = image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    return np.array([img_array])

# Function to generate voice summary
def generate_voice_summary(disease_name, summary_text, language="en"):
    try:
        tts = gtts.gTTS(text=f"{disease_name}. {summary_text}", lang=language, slow=False)
        audio_file = "disease_summary.mp3"
        tts.save(audio_file)
        
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode()
        
        if os.path.exists(audio_file):
            os.remove(audio_file)
            
        return audio_b64
    except Exception as e:
        st.error(f"Error generating voice: {e}")
        return None

# Class names
class_name = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Language code mapping for text-to-speech
language_to_tts_code = {
    "English": "en",
    "Hindi (हिन्दी)": "hi",
    "Tamil (தமிழ்)": "ta",
    "Telugu (తెలుగు)": "te",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Malayalam (മലയാളം)": "ml",
    "Bengali (বাংলা)": "bn",
    "Marathi (मराठी)": "mr",
    "Gujarati (ગુજરાતી)": "gu",
    "Punjabi (ਪੰਜਾਬੀ)": "pa",
    "Odia (ଓଡ଼ିଆ)": "or"
}

def main():
    with st.sidebar:
        selected_language = st.selectbox(
            "Select Language",
            options=list(languages.keys())
        )
        
        model_path = st.text_input(
            "Model Path",
            value='mod_1.keras',
            help="/content/trained_model0.keras"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info("Plant disease detection using deep learning")

    txt = languages[selected_language]
    st.title(txt["title"])
    st.write(txt["description"])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            txt["upload_prompt"],
            type=["jpg", "png", "jpeg"]
        )
    
    model = load_model(model_path)
    
    if model is None:
        st.error(txt["no_model"])
    
    if uploaded_file and model:
        try:
            image = Image.open(uploaded_file)
            with col1:
                st.image(image, caption=txt["uploaded_caption"])
            
            input_arr = preprocess_image(image)
            prediction = model.predict(input_arr)
            result_index = np.argmax(prediction)
            predicted_disease = class_name[result_index]
            confidence = float(prediction[0][result_index] * 100)
            
            with col2:
                st.markdown("### Results")
                st.success(f"{txt['prediction_text']} {predicted_disease}")
                st.progress(confidence/100)
                st.write(f"{txt['confidence']}: {confidence:.2f}%")
                
                st.markdown(f"### {txt['disease_summary']}")
                summary = get_disease_summary(predicted_disease, selected_language)
                st.write(summary)
                
                st.markdown(f"### {txt['play_voice']}")
                tts_lang = language_to_tts_code.get(selected_language, "en")
                display_name = predicted_disease.replace("___", " ")
                audio_b64 = generate_voice_summary(display_name, summary, tts_lang)
                if audio_b64:
                    st.audio(f"data:audio/mp3;base64,{audio_b64}", format="audio/mp3")
                
                st.markdown("### Top Predictions")
                top_indices = np.argsort(prediction[0])[-3:][::-1]
                for i, idx in enumerate(top_indices):
                    st.write(f"{i+1}. {class_name[idx]} ({prediction[0][idx]*100:.2f}%)")
                # Add optional visualization of model architecture
                if st.checkbox("Show Model Architecture"):
                    # Format model summary as text
                    stringlist = []
                    model.summary(print_fn=lambda x: stringlist.append(x))
                    model_summary = "\n".join(stringlist)
                    st.text(model_summary)
                
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()