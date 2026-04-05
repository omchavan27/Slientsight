import streamlit as st
import torch
import cv2
import numpy as np
from torchvision import transforms
from streamlit_option_menu import option_menu
from model_engine import load_model
from utils import ben_graham_preprocessing, generate_heatmap
from fpdf import FPDF
import tempfile
import datetime
import os

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="SilentSight AI", layout="wide", initial_sidebar_state="collapsed")

# --- 2. PDF GENERATION ENGINE (Vertical Stack) ---
def create_pdf(img_name, diagnosis, confidence, original, enhanced, heatmap):
    pdf = FPDF()
    
    # Page 1: Diagnostic Summary
    pdf.add_page()
    pdf.set_fill_color(37, 99, 235) # Blue header
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_font("Arial", 'B', 24)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(190, 20, "SILENTSIGHT AI REPORT", ln=True, align='C')
    
    pdf.set_y(45)
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(100, 10, f"Date Generated: {datetime.date.today().strftime('%B %d, %Y')}", ln=True)
    pdf.cell(100, 10, f"Source File: {img_name}", ln=True)
    
    pdf.ln(10)
    pdf.set_fill_color(240, 244, 248)
    pdf.rect(10, 75, 190, 35, 'F')
    pdf.set_y(80)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(190, 10, f"DIAGNOSIS: {diagnosis}", ln=True, align='C')
    pdf.set_font("Arial", '', 12)
    pdf.cell(190, 10, f"AI Confidence Score: {confidence}", ln=True, align='C')

    # Page 2: Vertical Image Evidence
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(37, 99, 235)
    pdf.cell(190, 10, "CLINICAL VISUAL EVIDENCE", ln=True, align='C')
    pdf.ln(5)

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = [os.path.join(tmpdir, f"img_{i}.jpg") for i in range(3)]
        cv2.imwrite(paths[0], cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
        cv2.imwrite(paths[1], cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
        cv2.imwrite(paths[2], cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))

        labels = ["1. Original Fundus Scan", "2. AI Enhanced View", "3. AI Lesion Localization (Heatmap)"]
        for i in range(3):
            pdf.set_font("Arial", 'B', 11)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(100, 8, labels[i], ln=True)
            pdf.image(paths[i], x=45, w=120)
            pdf.ln(10)
        
    return bytes(pdf.output())

# --- 3. CUSTOM STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #f8fafc; }
    .medical-card { background-color: white; border-radius: 1.5rem; border: 1px solid #e2e8f0; padding: 2rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); margin-bottom: 1.5rem; }
    .stButton>button { background-color: #2563eb !important; color: white !important; border-radius: 0.75rem !important; font-weight: 600 !important; width: 100%; border: none !important; transition: 0.3s; }
    .stButton>button:hover { background-color: #1d4ed8 !important; transform: translateY(-2px); }
    </style>
    """, unsafe_allow_html=True)

# --- 4. NAVIGATION ---
if 'menu_option' not in st.session_state:
    st.session_state.menu_option = 0

selected = option_menu(
    menu_title=None, 
    options=["Home", "Upload Image", "About"], 
    icons=["house", "cloud-upload", "info-circle"], 
    default_index=st.session_state.menu_option, 
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "white", "border-bottom": "1px solid #e2e8f0"},
        "nav-link-selected": {"background-color": "#eff6ff", "color": "#2563eb", "border-radius": "0.5rem"}
    }
)

# --- 5. APP LOGIC ---

if selected == "Home":
    st.markdown("""
        <div style="text-align: center; padding: 5rem 0 2rem 0;">
            <h1 style="font-size: 4rem; font-weight: 800; color: #0f172a;">Early Detection <br><span style="color: #2563eb;">Saves Vision.</span></h1>
            <p style="font-size: 1.2rem; color: #475569; margin: 1.5rem auto; max-width: 600px;">Diabetic Retinopathy detection using ResNet-50 architecture.</p>
        </div>
    """, unsafe_allow_html=True)
    
    _, btn_col, _ = st.columns([1.1, 0.4, 1])
    with btn_col:
        if st.button("Start Analysing"):
            st.session_state.menu_option = 1
            st.rerun()

elif selected == "Upload Image":
    st.session_state.menu_option = 1
    _, center_col, _ = st.columns([0.15, 0.7, 0.15])
    
    with center_col:
        st.markdown('<div class="medical-card">', unsafe_allow_html=True)
        st.subheader("Image Acquisition")
        uploaded_file = st.file_uploader("Upload fundus photograph", type=["jpg", "png", "jpeg"])
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file:
            # --- MODEL STABILITY FIX ---
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = load_model("best_model.pth")
            model.to(device)
            model.eval() # CRITICAL: Disables training-only layers
            
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # --- PREPROCESSING FIX ---
            # Resizing to 224 BEFORE Ben Graham ensures features are at the right scale for ResNet
            img_resized = cv2.resize(img, (224, 224))
            proc_img = ben_graham_preprocessing(img_resized)
            proc_rgb = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
            
            # --- INFERENCE FIX ---
            transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_tensor = transform(proc_img).unsqueeze(0).to(device)
            
            with torch.no_grad(): # Disables gradient tracking to ensure consistent output
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                confidence, pred = torch.max(probs, 1)
                
            diag_map = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}
            res_text = diag_map[pred.item()]
            conf_val = f"{confidence.item()*100:.2f}%"
            
            # Generate Visuals
            heat_img = generate_heatmap(model, input_tensor, proc_img)

            # --- UI: REPORT ---
            st.markdown('<div class="medical-card">', unsafe_allow_html=True)
            st.subheader("🧬 2. AI Diagnostic Report")
            c1, c2 = st.columns(2)
            c1.metric("DIAGNOSIS", res_text)
            c2.metric("CONFIDENCE", conf_val)
            
            pdf_bytes = create_pdf(uploaded_file.name, res_text, conf_val, img_rgb, proc_rgb, heat_img)
            st.download_button(label="📥 Download Clinical Report (PDF)", data=pdf_bytes, file_name=f"SilentSight_Report.pdf", mime="application/pdf")
            st.markdown('</div>', unsafe_allow_html=True)

            # --- UI: VISUALS ---
            st.markdown('<div class="medical-card">', unsafe_allow_html=True)
            st.subheader("🔍 3. Visual Analysis")
            t1, t2, t3 = st.tabs(["Original Scan", "Pre-processed View", "Localization Heatmap"])
            t1.image(img_rgb, use_container_width=True)
            t2.image(proc_rgb, use_container_width=True)
            t3.image(heat_img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

elif selected == "About":
    st.session_state.menu_option = 2
    st.markdown(""" 👁️ SilentSight – AI for Early Eye Disease Detection

SilentSight is an advanced AI-powered platform designed to assist in the *early detection of Diabetic Retinopathy* using retinal (fundus) images.

---

### 🚀 Our Mission
To make *eye disease screening fast, accessible, and affordable* using cutting-edge artificial intelligence, helping prevent vision loss worldwide.

---

### 🧠 How It Works
- Upload a retinal image  
- AI model analyzes the image  
- Detects severity level (Grade 0–4)  
- Generates visual explanation (Grad-CAM)  
- Provides a simple diagnostic report  

---

### ⚙️ Technology Used
- Deep Learning (CNN)  
- ResNet-50 Architecture  
- Computer Vision  
- Explainable AI (Grad-CAM)  
- Python & Streamlit  

---

### 📊 Dataset
APTOS 2019 Blindness Detection Dataset (Kaggle)

---

### 💡 Key Features
✔ Fast and accurate predictions  
✔ Explainable AI visualization  
✔ User-friendly interface  
✔ Downloadable medical report  

---

### ⚠️ Disclaimer
This tool is intended for *screening purposes only* and should not replace professional medical diagnosis.

---

### 👨‍💻 Developed By
Team Elite/ Om Chavan/ Sakshi Pardhi  

### 📧 Contact
silentsight@gmail.com""")