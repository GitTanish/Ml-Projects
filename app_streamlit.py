import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import base64
import json
import gtts

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

# Disease summaries database - English language
disease_summaries = {
    "Apple___Apple_scab": "Apple scab is a common fungal disease that affects apple trees, causing dark, scabby lesions on leaves and fruit. It thrives in cool, wet spring weather and can significantly reduce fruit quality and yield if left untreated.",
    "Apple___Black_rot": "Black rot is a fungal disease affecting apple trees that causes circular dark brown or black spots on leaves, and can lead to rotting of the fruit. It's more prevalent in humid weather and requires proper orchard management.",
    "Apple___Cedar_apple_rust": "Cedar apple rust is a fungal disease that requires both apple trees and juniper plants to complete its life cycle. It causes bright orange-yellow spots on apple leaves and can lead to premature leaf drop and reduced fruit yield.",
    "Apple___healthy": "This apple leaf is healthy with no signs of disease. Regular monitoring and proper care practices will help maintain plant health.",
    "Blueberry___healthy": "This blueberry plant shows healthy leaf characteristics with no visible disease symptoms. Regular monitoring and good agricultural practices will help maintain plant health.",
    "Cherry_(including_sour)___Powdery_mildew": "Powdery mildew on cherry trees appears as a white, powdery coating on leaves and sometimes fruit. It thrives in high humidity with moderate temperatures and can reduce fruit quality and yield if severe.",
    "Cherry_(including_sour)___healthy": "This cherry leaf displays healthy characteristics with no visible disease symptoms. Continue with regular monitoring and appropriate care practices.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Gray leaf spot is a significant fungal disease in corn, characterized by rectangular, grayish-brown lesions on leaves. It flourishes in humid conditions and can severely impact crop yields if not managed properly.",
    "Corn_(maize)___Common_rust_": "Common rust in corn appears as small, circular to elongated, brown to reddish-brown pustules on both leaf surfaces. It can reduce photosynthesis and yield in severe cases, especially in sweet corn varieties.",
    "Corn_(maize)___Northern_Leaf_Blight": "Northern Leaf Blight causes long, elliptical, grayish-green to brown lesions on corn leaves. In severe cases, it can significantly reduce crop yield. It's most problematic in warm, humid conditions with heavy dew.",
    "Corn_(maize)___healthy": "This corn leaf is showing normal, healthy characteristics with no visible disease symptoms. Continue regular monitoring for optimal plant health.",
    "Grape___Black_rot": "Black rot is a serious fungal disease affecting grapes, causing circular, dark lesions on leaves and rotting of the fruit. It's most active in warm, humid weather and can devastate an entire crop if left untreated.",
    "Grape___Esca_(Black_Measles)": "Esca, or Black Measles, is a complex fungal disease affecting grapevines, causing tiger-striped leaf patterns and black spotting on grapes. It can lead to reduced fruit quality and even vine death in severe cases.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Isariopsis Leaf Spot causes small, dark, angular to circular lesions on grape leaves. It thrives in warm, wet conditions and can cause premature defoliation, reducing fruit quality and vineyard productivity.",
    "Grape___healthy": "This grape leaf shows healthy characteristics with no visible disease symptoms. Regular monitoring and proper vineyard management will help maintain plant health.",
    "Orange___Haunglongbing_(Citrus_greening)": "Huanglongbing, or Citrus Greening, is a serious bacterial disease spread by psyllids that causes mottled leaves, stunted growth, and bitter, misshapen fruit. It's one of the most devastating citrus diseases worldwide with no cure.",
     "Peach___Bacterial_spot": "Bacterial spot in peaches is caused by Xanthomonas campestris pv. pruni. It leads to dark, sunken lesions on leaves, fruit, and twigs, often resulting in premature leaf drop and fruit blemishes. The disease thrives in warm, wet conditions and can significantly reduce fruit quality and yield.",
    "Peach___healthy": "This peach leaf appears healthy with no visible signs of disease. Maintaining proper orchard hygiene and monitoring can help sustain plant health.",
    "Pepper,_bell___Bacterial_spot": "Bacterial spot in bell peppers is caused by Xanthomonas campestris pv. vesicatoria. It manifests as small, water-soaked spots on leaves and fruit, which may enlarge and become necrotic. The disease is favored by high humidity and warm temperatures, potentially leading to defoliation and fruit drop.",
    "Pepper,_bell___healthy": "This bell pepper plant shows no signs of disease. Regular inspection and proper cultural practices can help maintain its health.",
    "Potato___Early_blight": "Early blight in potatoes is caused by the fungus Alternaria solani. It presents as dark, concentric spots on older leaves, leading to defoliation. The disease thrives in warm, humid conditions and can reduce tuber yield if not managed properly.",
    "Potato___Late_blight": "Late blight, caused by Phytophthora infestans, is a serious disease in potatoes. It causes water-soaked lesions on leaves and stems, which rapidly turn brown and necrotic. Infected tubers develop firm, brown decay. The disease spreads quickly in cool, moist conditions.",
    "Potato___healthy": "This potato plant is healthy with no visible disease symptoms. Consistent monitoring and proper cultivation practices are essential for maintaining plant health.",
    "Raspberry___healthy": "This raspberry plant shows healthy foliage with no signs of disease. Regular care and monitoring are key to sustaining its health.",
    "Soybean___healthy": "This soybean plant appears healthy, exhibiting no disease symptoms. Proper crop rotation and field management contribute to its vigor.",
    "Squash___Powdery_mildew": "Powdery mildew in squash is caused by various fungi, leading to white, powdery spots on leaves and stems. It can cause leaf curling, yellowing, and premature defoliation, affecting fruit development. The disease favors dry days and humid nights.",
    "Strawberry___Leaf_scorch": "Leaf scorch in strawberries is caused by the fungus Diplocarpon earlianum. Symptoms include small, purple spots on leaves that enlarge and coalesce, leading to browning and leaf death. Severe infections can reduce plant vigor and fruit yield.",
    "Strawberry___healthy": "This strawberry plant displays healthy leaves and no signs of disease. Maintaining good air circulation and sanitation helps prevent common strawberry diseases.",
    "Tomato___Bacterial_spot": "Bacterial spot in tomatoes, caused by Xanthomonas spp., results in small, dark, water-soaked lesions on leaves, stems, and fruit. Severe infections can lead to defoliation and fruit blemishes, reducing marketability.",
    "Tomato___Early_blight": "Early blight in tomatoes is caused by Alternaria solani. It appears as dark spots with concentric rings on older leaves, leading to yellowing and defoliation. The disease can also affect stems and fruit, impacting yield.",
    "Tomato___Late_blight": "Late blight, caused by Phytophthora infestans, is a devastating disease in tomatoes. It causes large, water-soaked lesions on leaves and stems, which rapidly turn brown. Infected fruit develop firm, dark rot. The disease spreads rapidly in cool, wet conditions.",
    "Tomato___Leaf_Mold": "Leaf mold in tomatoes is caused by the fungus Passalora fulva. It presents as pale green to yellow spots on upper leaf surfaces, with corresponding olive-green to gray mold on the undersides. High humidity and poor air circulation favor its development.",
    "Tomato___Septoria_leaf_spot": "Septoria leaf spot, caused by Septoria lycopersici, leads to small, circular spots with dark borders and gray centers on leaves. It typically starts on lower leaves and progresses upward, causing defoliation and reduced yield.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Two-spotted spider mites are tiny pests that feed on tomato plants, causing stippling and yellowing of leaves. Severe infestations can lead to webbing and defoliation, stressing the plant and reducing yield.",
    "Tomato___Target_Spot": "Target spot in tomatoes is caused by Corynespora cassiicola. It results in circular lesions with concentric rings on leaves, stems, and fruit. The disease can cause significant defoliation and fruit spotting, impacting yield and quality.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato Yellow Leaf Curl Virus (TYLCV) is transmitted by whiteflies and causes stunted growth, upward leaf curling, and yellowing. Infected plants produce little to no fruit, leading to significant yield losses.",
    "Tomato___Tomato_mosaic_virus": "Tomato Mosaic Virus (ToMV) causes mottled light and dark green patterns on leaves, leaf curling, and reduced fruit size and quality. The virus is mechanically transmitted and can persist in plant debris and contaminated tools.",
    "Tomato___healthy": "This tomato plant is healthy with vibrant green leaves and no signs of disease. Regular monitoring and good cultural practices help maintain plant health."
}

# Function to load model
@st.cache_resource
def load_model(model_path):
    try:
        if os.path.exists(model_path):
            return tf.keras.models.load_model(model_path)
        else:
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to preprocess image exactly as in the test file
def preprocess_image(image, target_size=(128, 128)):
    img = image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    return np.array([img_array])  # Convert single image to a batch

# Function to generate voice summary
def generate_voice_summary(disease_name, summary_text, language="en"):
    try:
        tts = gtts.gTTS(text=f"{disease_name}. {summary_text}", lang=language, slow=False)
        audio_file = "disease_summary.mp3"
        tts.save(audio_file)
        
        # Read the audio file and get base64 encoding for HTML playback
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode()
        
        # Clean up the file
        if os.path.exists(audio_file):
            os.remove(audio_file)
            
        return audio_b64
    except Exception as e:
        st.error(f"Error generating voice: {e}")
        return None

# Class names - fixed the format to match the test file
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
    # Sidebar for language selection
    with st.sidebar:
        selected_language = st.selectbox(
            "Select Language / भाषा चुने ",
            options=list(languages.keys())
        )
        
        # Model path can be configurable (optional)
        model_path = st.text_input(
            "Model Path", 
            value='/content/plant_disease_model_1.keras',
            help="Path to your trained model file"
        )
        
        # Display some information about the app
        st.markdown("---")
        st.markdown("### About")
        st.info(
            "This application uses a deep learning model to detect "
            "diseases in plant leaves from uploaded images. It can also "
            "provide voice summaries about detected diseases."
        )
    
    # Get text based on selected language
    txt = languages[selected_language]
    
    # Main content
    st.title(txt["title"])
    st.write(txt["description"])
    
    # Create columns for better layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # File uploader
        uploaded_file = st.file_uploader(
            txt["upload_prompt"], 
            type=["jpg", "png", "jpeg"],
            help="Supported formats: JPG, JPEG, PNG"
        )
    
    # Load the model
    model = load_model(model_path)
    
    if model is None:
        st.error(txt["no_model"])
    
    if uploaded_file is not None and model is not None:
        try:
            # Display uploaded image
            image = Image.open(uploaded_file)
            with col1:
                st.image(image, caption=txt["uploaded_caption"], use_column_width=True)
            
            # Preprocess image exactly as in test file
            input_arr = preprocess_image(image)
            
            # Predict
            prediction = model.predict(input_arr)
            result_index = np.argmax(prediction)
            predicted_disease = class_name[result_index]
            confidence = float(prediction[0][result_index] * 100)
            
            # Display results
            with col2:
                st.markdown("### Results")
                st.success(f"{txt['prediction_text']} **{predicted_disease}**")
                st.progress(confidence/100)
                st.write(f"{txt['confidence']}: {confidence:.2f}%")
                
                # Show disease summary
                st.markdown(f"### {txt['disease_summary']}")
                
                # Get disease summary from the database (default to English for now)
                # In a full implementation, you would translate these summaries for each language
                disease_summary = disease_summaries.get(
                    predicted_disease, 
                    "Detailed information about this disease is not available at the moment."
                )
                st.write(disease_summary)
                
                # Voice playback option
                st.markdown("### " + txt['play_voice'])
                
                # Get TTS language code based on selected language
                tts_lang = language_to_tts_code.get(selected_language, "en")
                
                # Generate voice and create audio player
                audio_b64 = generate_voice_summary(
                    predicted_disease.replace("___", " ").replace("_", " "), 
                    disease_summary,
                    tts_lang
                )
                
                if audio_b64:
                    st.audio(f"data:audio/mp3;base64,{audio_b64}", format="audio/mp3")
                else:
                    st.warning("Voice generation failed. Please try again.")
                
                # Show top 3 predictions
                st.markdown("### Top 3 Predictions")
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
            st.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main()