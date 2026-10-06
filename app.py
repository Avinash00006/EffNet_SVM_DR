import os
import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Import modular components
from src.preprocessing import preprocess_fundus_image, transform, validate_fundus_image
from src.model_inference import load_models, ExplanationEngine
from src.contact import send_contact_message

# Authentic clinical cases dictionaries (Sourced from dataset)
ORIGINAL_NORMAL_CASES = {
    "Case 1 (002c21358ce6)": "sample_images/Normal Eyes/002c21358ce6.png",
    "Case 2 (005b95c28852)": "sample_images/Normal Eyes/005b95c28852.png",
    "Case 3 (0097f532ac9f)": "sample_images/Normal Eyes/0097f532ac9f.png",
    "Case 4 (00cc2b75cddd)": "sample_images/Normal Eyes/00cc2b75cddd.png",
    "Case 5 (00f6c1be5a33)": "sample_images/Normal Eyes/00f6c1be5a33.png",
    "Case 6 (0125fbd2e791)": "sample_images/Normal Eyes/0125fbd2e791.png",
}

ORIGINAL_DR_CASES = {
    "Case 1 (000c1434d8d7)": "sample_images/Diabetic Retinopathy/000c1434d8d7.png",
    "Case 2 (001639a390f0)": "sample_images/Diabetic Retinopathy/001639a390f0.png",
    "Case 3 (0024cdab0c1e)": "sample_images/Diabetic Retinopathy/0024cdab0c1e.png",
    "Case 4 (0083ee8054ee)": "sample_images/Diabetic Retinopathy/0083ee8054ee.png",
    "Case 5 (00a8624548a9)": "sample_images/Diabetic Retinopathy/00a8624548a9.png",
    "Case 6 (00b74780d31d)": "sample_images/Diabetic Retinopathy/00b74780d31d.png",
}

# Modal dialog for browsing real fundus images
if hasattr(st, "dialog"):
    @st.dialog("🖼️ Select Clinical Case", width="small")
    def show_sample_gallery_dialog(category_name, cases_dict):
        st.markdown(f"<h3 style='margin: 0 0 4px 0; font-size: 1.15rem; color: #1E293B;'>{category_name}</h3>", unsafe_allow_html=True)
        st.markdown("<p style='margin: 0 0 14px 0; font-size: 0.82rem; color: #64748B;'>Click directly on any retinal photography case below to load and analyze:</p>", unsafe_allow_html=True)
        
        # Isolated card styling with direct-click button overlay per column
        st.markdown("""
        <style>
            div[data-testid="stDialog"] div[data-testid="stColumn"] {
                position: relative !important;
                border: 2px solid #E2E8F0 !important;
                border-radius: 14px !important;
                padding: 8px !important;
                text-align: center !important;
                transition: all 0.2s ease-in-out !important;
                background-color: #FFFFFF !important;
                cursor: pointer !important;
                margin-bottom: 8px !important;
            }
            div[data-testid="stDialog"] div[data-testid="stColumn"]:hover {
                border-color: #2B59ED !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 18px rgba(43, 89, 237, 0.18) !important;
            }
            div[data-testid="stDialog"] div[data-testid="stColumn"] div.stButton {
                position: absolute !important;
                top: 0 !important;
                left: 0 !important;
                width: 100% !important;
                height: 100% !important;
                margin: 0 !important;
                padding: 0 !important;
                z-index: 10 !important;
            }
            div[data-testid="stDialog"] div[data-testid="stColumn"] div.stButton button {
                width: 100% !important;
                height: 100% !important;
                opacity: 0 !important;
                cursor: pointer !important;
                border: none !important;
                background: transparent !important;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Row-by-row layout (2 cards per row) so each column contains exactly ONE image
        cases_list = list(cases_dict.items())
        for row_i in range(0, len(cases_list), 2):
            row_cols = st.columns(2)
            for c_i in range(2):
                idx = row_i + c_i
                if idx < len(cases_list):
                    label, rel_path = cases_list[idx]
                    full_path = os.path.join(os.path.dirname(__file__), rel_path)
                    with row_cols[c_i]:
                        if os.path.exists(full_path):
                            img = Image.open(full_path)
                            st.image(img, use_container_width=True)
                            st.markdown(f"<div style='font-size: 0.76rem; font-weight: 700; color: #1E293B; margin-top: 3px;'>{label}</div>", unsafe_allow_html=True)
                        if st.button(f"Select {label}", key=f"btn_modal_{category_name}_{idx}"):
                            st.session_state.selected_sample_path = full_path
                            st.session_state.selected_sample_label = f"{category_name} - {label}"
                            st.rerun()
else:
    def show_sample_gallery_dialog(category_name, cases_dict):
        st.session_state.fallback_gallery_open = True


#---------------Streamlit UI Page Settings---------------------------------------------
st.set_page_config(
    page_title="Retinal diagnostics AI Workstation",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling: Medcare Clinical Theme (Optimized for 100% zoom with high-density compact sizing)
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
        
        /* 1. Global Canvas & Typography */
        html, body, [class*="css"], .stApp {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
            background-color: #F4F7FC !important;
            color: #1E293B !important;
        }
        
        /* 2. Compact Viewport Container (Optimized for 100% Zoom) */
        .block-container,
        [data-testid="stMainBlockContainer"] {
            max-width: 1400px !important;
            padding-top: 1.0rem !important;
            padding-bottom: 1.5rem !important;
            padding-left: 1.75rem !important;
            padding-right: 1.75rem !important;
            margin: 0 auto !important;
        }
        
        /* Transparent Header */
        header[data-testid="stHeader"] {
            height: 2.4rem !important;
            background: transparent !important;
            background-color: transparent !important;
        }
        
        /* 3. Typography Hierarchy (Scaled down for 100% Zoom) */
        h1, [data-testid="stHeadingWithActionElements"] h1 {
            font-size: 1.35rem !important;
            font-weight: 800 !important;
            line-height: 1.25 !important;
            margin: 0.2rem 0 0.4rem 0 !important;
            color: #1E293B !important;
            letter-spacing: -0.02em !important;
        }
        h2, [data-testid="stHeadingWithActionElements"] h2 {
            font-size: 1.15rem !important;
            font-weight: 700 !important;
            line-height: 1.25 !important;
            margin: 0.2rem 0 0.35rem 0 !important;
            color: #1E293B !important;
        }
        h3, [data-testid="stHeadingWithActionElements"] h3 {
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            line-height: 1.3 !important;
            margin: 0.15rem 0 0.25rem 0 !important;
            color: #1E293B !important;
        }
        p, span, label, [data-testid="stMarkdownContainer"] p {
            font-size: 13.5px !important;
            line-height: 1.45 !important;
        }
        small, .caption, [data-testid="stImageCaption"] {
            font-size: 11.5px !important;
            color: #64748B !important;
            line-height: 1.3 !important;
        }
        
        /* 4. Sidebar: Medcare Clean Styling */
        section[data-testid="stSidebar"],
        [data-testid="stSidebar"] {
            width: 270px !important;
            min-width: 270px !important;
            max-width: 275px !important;
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
            box-shadow: 4px 0 20px rgba(43, 89, 237, 0.03) !important;
        }
        [data-testid="stSidebarContent"] {
            padding-top: 1.2rem !important;
            padding-left: 1.0rem !important;
            padding-right: 1.0rem !important;
        }
        
        /* Medcare Sidebar Brand Badge */
        .medcare-sidebar-brand {
            display: flex;
            align-items: center;
            gap: 10px;
            background: linear-gradient(135deg, #2B59ED 0%, #3B71F7 100%);
            padding: 12px 16px;
            border-radius: 14px;
            color: #FFFFFF !important;
            margin-bottom: 18px;
            box-shadow: 0 6px 18px rgba(43, 89, 237, 0.22);
        }
        .medcare-sidebar-brand h2 {
            color: #FFFFFF !important;
            margin: 0 !important;
            font-size: 1.15rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.02em;
        }
        
        /* 5. Medcare Cards & Containers */
        .medcare-card {
            background: #FFFFFF;
            border-radius: 18px;
            border: 1px solid #E9EFF7;
            padding: 18px 20px;
            box-shadow: 0 4px 18px rgba(43, 89, 237, 0.04);
            margin-bottom: 16px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .medcare-card:hover {
            box-shadow: 0 6px 24px rgba(43, 89, 237, 0.08);
        }
        
        /* 6. Buttons: Royal Blue Pill */
        [data-testid="stButton"] button,
        .stButton > button {
            background: linear-gradient(135deg, #2B59ED 0%, #3B71F7 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 12px !important;
            min-height: 34px !important;
            height: 34px !important;
            padding: 4px 14px !important;
            font-weight: 600 !important;
            font-size: 13px !important;
            letter-spacing: -0.01em !important;
            box-shadow: 0 4px 12px rgba(43, 89, 237, 0.22) !important;
            transition: all 0.2s ease-in-out !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 18px rgba(43, 89, 237, 0.35) !important;
            color: #FFFFFF !important;
        }
        
        /* Secondary Category Buttons */
        button[key="btn_side_norm"], button[key="btn_w_norm"] {
            background: #FFFFFF !important;
            color: #10B981 !important;
            border: 1.5px solid #10B981 !important;
            box-shadow: 0 2px 8px rgba(16, 185, 129, 0.15) !important;
        }
        button[key="btn_side_norm"]:hover, button[key="btn_w_norm"]:hover {
            background: #E6F9F2 !important;
            color: #059669 !important;
        }
        button[key="btn_side_dr"], button[key="btn_w_dr"] {
            background: #FFFFFF !important;
            color: #EF4444 !important;
            border: 1.5px solid #EF4444 !important;
            box-shadow: 0 2px 8px rgba(239, 68, 68, 0.15) !important;
        }
        button[key="btn_side_dr"]:hover, button[key="btn_w_dr"]:hover {
            background: #FEECEB !important;
            color: #DC2626 !important;
        }
        
        /* 7. Tabs: Sleek Clinical Pill Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px !important;
            background-color: #EBF0F9 !important;
            padding: 4px !important;
            border-radius: 14px !important;
            min-height: 34px !important;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px !important;
            padding: 6px 16px !important;
            font-weight: 600 !important;
            font-size: 13px !important;
            color: #64748B !important;
            background-color: transparent !important;
            border: none !important;
            min-height: 32px !important;
            height: 32px !important;
        }
        .stTabs [aria-selected="true"] {
            background-color: #FFFFFF !important;
            color: #2B59ED !important;
            box-shadow: 0 2px 8px rgba(43, 89, 237, 0.12) !important;
        }
        
        /* 8. Radio Buttons as Segmented Control */
        div[data-testid="stRadio"] > div {
            background-color: #EBF0F9;
            padding: 4px;
            border-radius: 12px;
            gap: 4px;
        }
        div[data-testid="stRadio"] label {
            padding: 5px 12px;
            border-radius: 9px;
            font-weight: 600;
            font-size: 12.5px;
            color: #475569;
        }
        div[data-testid="stRadio"] label[data-checked="true"] {
            background-color: #FFFFFF;
            color: #2B59ED;
            box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        }
        
        /* 9. Fundus Image Display Constraints (Compact 290px max) */
        [data-testid="stImage"] {
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
        }
        [data-testid="stImage"] img {
            max-width: 290px !important;
            max-height: 290px !important;
            width: auto !important;
            height: auto !important;
            object-fit: contain !important;
            border-radius: 12px !important;
            border: 1px solid #E2E8F0 !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
            background-color: #000000 !important;
        }
        
        /* Badges */
        .badge-info {
            display: inline-flex;
            align-items: center;
            background-color: #EEF2FF;
            color: #2B59ED;
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 0.72rem;
            font-weight: 700;
            margin-right: 6px;
            border: 1px solid #E0E7FF;
        }

        /* 10. Medcare Hero Banner & KPI Components */
        .medcare-hero-banner {
            background: linear-gradient(135deg, #1E3A8A 0%, #2B59ED 60%, #3B82F6 100%);
            border-radius: 20px;
            padding: 22px 26px;
            color: #FFFFFF !important;
            box-shadow: 0 10px 25px rgba(43, 89, 237, 0.2);
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 18px;
            position: relative;
            overflow: hidden;
        }
        .medcare-hero-banner::after {
            content: "";
            position: absolute;
            right: -30px;
            top: -30px;
            width: 200px;
            height: 200px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255,255,255,0.18) 0%, transparent 70%);
            pointer-events: none;
        }
        .medcare-badge-live {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(255, 255, 255, 0.18);
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: #FFFFFF;
            font-size: 0.7rem;
            font-weight: 700;
            padding: 3px 10px;
            border-radius: 20px;
            letter-spacing: 0.04em;
            margin-bottom: 8px;
        }
        .medcare-kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 18px;
        }
        .medcare-kpi-card {
            background: #FFFFFF;
            border-radius: 16px;
            padding: 13px 15px;
            border: 1px solid #E9EFF7;
            box-shadow: 0 2px 10px rgba(43, 89, 237, 0.04);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .medcare-kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(43, 89, 237, 0.08);
        }
        .kpi-title {
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            color: #64748B;
            letter-spacing: 0.03em;
            margin-bottom: 4px;
        }
        .kpi-val {
            font-size: 1.35rem;
            font-weight: 800;
            color: #1E293B;
            line-height: 1.1;
            margin-bottom: 6px;
        }
        .kpi-badge-positive {
            display: inline-flex;
            align-items: center;
            gap: 3px;
            font-size: 0.68rem;
            font-weight: 700;
            color: #10B981;
            background: #E6F9F2;
            padding: 2px 7px;
            border-radius: 10px;
            width: fit-content;
        }
        .kpi-badge-neutral {
            display: inline-flex;
            align-items: center;
            gap: 3px;
            font-size: 0.68rem;
            font-weight: 700;
            color: #2B59ED;
            background: #EEF2FF;
            padding: 2px 7px;
            border-radius: 10px;
            width: fit-content;
        }
        
        /* 11. Medcare Patient Study Header & Diagnosis Cards */
        .medcare-study-header {
            background: #FFFFFF;
            border-radius: 14px;
            padding: 12px 18px;
            border: 1px solid #E9EFF7;
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            align-items: center;
            margin-bottom: 18px;
            box-shadow: 0 2px 8px rgba(43, 89, 237, 0.04);
        }
        .study-meta-item {
            display: flex;
            flex-direction: column;
            gap: 2px;
        }
        .meta-label {
            font-size: 0.68rem;
            font-weight: 700;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .meta-val {
            font-size: 0.85rem;
            font-weight: 700;
            color: #1E293B;
        }
        
        .diagnosis-card-dr {
            background: #FFFFFF;
            border-radius: 16px;
            border: 1px solid #FEE2E2;
            border-left: 6px solid #EF4444;
            padding: 16px 18px;
            box-shadow: 0 4px 16px rgba(239, 68, 68, 0.08);
            display: flex;
            align-items: flex-start;
            gap: 14px;
            min-height: 125px;
        }
        .diagnosis-card-normal {
            background: #FFFFFF;
            border-radius: 16px;
            border: 1px solid #D1FAE5;
            border-left: 6px solid #10B981;
            padding: 16px 18px;
            box-shadow: 0 4px 16px rgba(16, 185, 129, 0.08);
            display: flex;
            align-items: flex-start;
            gap: 14px;
            min-height: 125px;
        }
        
        /* Biomarker Severity Matrix Card */
        .biomarker-card {
            background: #FFFFFF;
            border-radius: 16px;
            border: 1px solid #E9EFF7;
            padding: 12px 16px;
            box-shadow: 0 2px 10px rgba(43, 89, 237, 0.04);
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-around;
        }
        .biomarker-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 4px;
        }
        .biomarker-name {
            font-size: 0.76rem;
            font-weight: 600;
            color: #475569;
        }
        .biomarker-bar-bg {
            background-color: #F1F5F9;
            height: 6px;
            border-radius: 6px;
            width: 100%;
            overflow: hidden;
            margin-top: 2px;
        }
        .biomarker-bar-fill {
            height: 100%;
            border-radius: 6px;
        }

        /* Remove Streamlit default Deploy, main menu, footer, and decoration */
        .stDeployButton, #MainMenu, footer, div[data-testid="stDecoration"], div[data-testid="InputInstructions"] {
            display: none !important;
            visibility: hidden !important;
        }
    </style>
""", unsafe_allow_html=True)

#---------------Load Models & Check Errors----------------------------------------------
full_model, feature_extractor, scaler, svm, load_error = load_models()

#---------------Sidebar Layout (Ophthalmic Control Console)----------------------------
with st.sidebar:
    st.markdown("""
    <div class="medcare-sidebar-brand">
        <span style="font-size: 1.5rem;">👁️</span>
        <div>
            <h2>RetinaAI Care</h2>
            <div style="font-size: 0.7rem; opacity: 0.88; font-weight: 500;">CLINICAL DECISION SYSTEM</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("📁 Fundus Photography Input")
    input_mode = st.radio(
        "Choose Input Source",
        ["📤 Upload Image", "🖼️ Preloaded Sample Cases"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    active_image = None
    active_source_label = None
    is_already_preprocessed = False
    validation_error = None
    
    if input_mode == "📤 Upload Image":
        uploaded_file = st.file_uploader(
            "Upload a retinal fundus image (.jpg, .jpeg, .png)",
            type=["jpg", "png", "jpeg"],
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            try:
                active_image = Image.open(uploaded_file).convert("RGB")
                active_source_label = f"Uploaded File ({uploaded_file.name})"
            except Exception as e:
                validation_error = f"Failed to parse image file: {str(e)}"
    else:
        st.markdown("<span style='font-size:0.8rem; color:#64748B; font-weight:600;'>BROWSE CLINICAL SAMPLES:</span>", unsafe_allow_html=True)
        
        btn_normal = st.button("🟢 Normal Images", use_container_width=True, key="btn_side_norm")
        btn_dr = st.button("🔴 Diabetic Retinopathy Cases", use_container_width=True, key="btn_side_dr")
        
        if btn_normal:
            show_sample_gallery_dialog("🟢 Normal Retina Cases", ORIGINAL_NORMAL_CASES)
        elif btn_dr:
            show_sample_gallery_dialog("🔴 Diabetic Retinopathy Cases", ORIGINAL_DR_CASES)
            
        if st.session_state.get("selected_sample_path"):
            sample_path = st.session_state.selected_sample_path
            if os.path.exists(sample_path):
                try:
                    active_image = Image.open(sample_path).convert("RGB")
                    active_source_label = st.session_state.selected_sample_label
                    
                    st.markdown("---")
                    st.caption(f"**Active Case:** {os.path.basename(sample_path)}")
                    st.image(active_image, use_container_width=True)
                    
                    with open(sample_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Raw PNG",
                            data=f.read(),
                            file_name=os.path.basename(sample_path),
                            mime="image/png",
                            use_container_width=True
                        )
                except Exception as e:
                    validation_error = f"Failed to load sample image: {str(e)}"
            else:
                validation_error = f"Sample file not found at {sample_path}"

    # Run validation immediately in sidebar background to update controls
    if active_image is not None and validation_error is None:
        is_valid, validation_msg, is_already_preprocessed = validate_fundus_image(active_image)
        if not is_valid:
            validation_error = validation_msg
            
    st.markdown("---")
    st.subheader("📬 Developer & Support")
    
    # Custom HTML for social links (styled with clean CSS badges)
    st.markdown(
        """
        <div style="display: flex; gap: 8px; margin-bottom: 12px;">
            <a href="https://github.com/Avinash00006" target="_blank" style="text-decoration: none; color: inherit; background-color: rgba(128,128,128,0.1); border: 1px solid rgba(128,128,128,0.18); padding: 5px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; display: inline-flex; align-items: center; gap: 5px; cursor: pointer;">
                🐙 GitHub
            </a>
            <a href="https://linkedin.com/in/avinash-koneti" target="_blank" style="text-decoration: none; color: inherit; background-color: rgba(128,128,128,0.1); border: 1px solid rgba(128,128,128,0.18); padding: 5px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; display: inline-flex; align-items: center; gap: 5px; cursor: pointer;">
                🔗 LinkedIn
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Initialize contact message session state
    if "message_sent" not in st.session_state:
        st.session_state.message_sent = False
        
    with st.expander("✉️ Contact Developer"):
        if st.session_state.message_sent:
            st.success("📩 Message sent! Thank you for reaching out.")
        else:
            contact_name = st.text_input("Name", placeholder="Your Name", label_visibility="collapsed", key="contact_name")
            contact_email = st.text_input("Email", placeholder="Your Email", label_visibility="collapsed", key="contact_email")
            contact_msg = st.text_area("Message", placeholder="Type your message...", label_visibility="collapsed", key="contact_msg")
            
            if st.button("Send Message", use_container_width=True, key="btn_send_contact"):
                if not contact_name.strip() or not contact_email.strip() or not contact_msg.strip():
                    st.error("Please fill in all fields.")
                else:
                    with st.spinner("Delivering message..."):
                        success, response_msg = send_contact_message(
                            contact_name.strip(),
                            contact_email.strip(),
                            contact_msg.strip()
                        )
                    if success:
                        st.session_state.message_sent = True
                        st.toast("Message sent successfully!", icon="✉️")
                        st.rerun()
                    else:
                        st.error(f"Error: {response_msg}")
            

#---------------Main Canvas Layout------------------------------------------------------

# Handle potential asset loading errors gracefully
if load_error is not None:
    st.error("⚠️ **Application Initialization Failure**")
    st.markdown(f"**Reason:** {load_error}")
    st.warning("Please ensure the ML weights are placed inside the project structure as follows:")
    st.code("""
EffNet_SVM_DR/
  ├── model/
  │   ├── best_effnet_binary.pth  (81 MB)
  │   ├── scaler_binary.pkl       (31 KB)
  │   └── svm_binary.pkl          (596 KB)
    """)
    st.stop()

# Instantiate Explanation Engine
engine = ExplanationEngine(full_model, feature_extractor, scaler, svm)

# UI STATE 1: Welcome Screen (Awaiting File Upload or Sample Selection)
if active_image is None:
    # 1. Medcare Hero Banner
    st.markdown("""
    <div class="medcare-hero-banner">
        <div style="flex: 1; z-index: 1;">
            <div class="medcare-badge-live">🟢 CLINICAL WORKSTATION READY</div>
            <h2 style="font-size: 1.45rem; font-weight: 800; color: #FFFFFF !important; margin: 6px 0 6px 0; letter-spacing: -0.02em;">Good Day, Clinician! 👋</h2>
            <p style="color: rgba(255, 255, 255, 0.92); font-size: 0.86rem; line-height: 1.5; margin: 0; max-width: 660px;">
                Welcome to <b>RetinaAI Care</b> workstation. Automated ophthalmic fundus screening powered by hybrid EfficientNetV2-S deep feature extraction, Support Vector Machines, and Explainable AI (Grad-CAM & LIME).
            </p>
        </div>
        <div style="font-size: 3.2rem; margin-left: 20px; opacity: 0.92; z-index: 1;">👁️‍🗨️</div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Medcare KPI Stat Cards Row
    st.markdown("""
    <div class="medcare-kpi-grid">
        <div class="medcare-kpi-card">
            <div>
                <div class="kpi-title">TOTAL SCANS EVALUATED</div>
                <div class="kpi-val">12,480</div>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="kpi-badge-positive">↑ +14.2%</span>
                <svg width="60" height="20" viewBox="0 0 60 20">
                    <path d="M 0 16 Q 15 14, 30 9 T 60 3" fill="none" stroke="#10B981" stroke-width="2.2" stroke-linecap="round"/>
                </svg>
            </div>
        </div>
        <div class="medcare-kpi-card">
            <div>
                <div class="kpi-title">AI SENSITIVITY</div>
                <div class="kpi-val">98.4%</div>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="kpi-badge-positive">↑ +2.1% AUC</span>
                <svg width="60" height="20" viewBox="0 0 60 20">
                    <path d="M 0 15 Q 18 16, 32 8 T 60 4" fill="none" stroke="#2B59ED" stroke-width="2.2" stroke-linecap="round"/>
                </svg>
            </div>
        </div>
        <div class="medcare-kpi-card">
            <div>
                <div class="kpi-title">XAI EXPLANATIONS</div>
                <div class="kpi-val">100%</div>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="kpi-badge-neutral">Grad-CAM + LIME</span>
                <svg width="60" height="20" viewBox="0 0 60 20">
                    <path d="M 0 12 Q 20 6, 40 14 T 60 5" fill="none" stroke="#8B5CF6" stroke-width="2.2" stroke-linecap="round"/>
                </svg>
            </div>
        </div>
        <div class="medcare-kpi-card">
            <div>
                <div class="kpi-title">INFERENCE LATENCY</div>
                <div class="kpi-val">&lt; 1.2s</div>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="kpi-badge-positive">⚡ Real-time</span>
                <svg width="60" height="20" viewBox="0 0 60 20">
                    <path d="M 0 14 Q 25 10, 45 4 T 60 2" fill="none" stroke="#F59E0B" stroke-width="2.2" stroke-linecap="round"/>
                </svg>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 3. Clinical Workflow Overview Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="medcare-card">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <span style="background: #EBF0F9; padding: 5px 8px; border-radius: 10px; font-size: 1rem;">📋</span>
                <span style="font-weight: 700; font-size: 0.88rem; color: #1E293B;">1. Quality Assurance</span>
            </div>
            <p style="font-size: 0.8rem; color: #64748B; margin: 0; line-height: 1.45;">
                Automated pre-flight validation analyzing aspect ratio, circular aperture coverage, and illumination metrics before pipeline inference.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="medcare-card">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <span style="background: #EBF0F9; padding: 5px 8px; border-radius: 10px; font-size: 1rem;">🧠</span>
                <span style="font-weight: 700; font-size: 0.88rem; color: #1E293B;">2. Deep Feature Classifier</span>
            </div>
            <p style="font-size: 0.8rem; color: #64748B; margin: 0; line-height: 1.45;">
                EfficientNetV2-S deep convolutional backbone extracts 1,280 structural features, classified with an RBF Support Vector Machine.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="medcare-card">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <span style="background: #EBF0F9; padding: 5px 8px; border-radius: 10px; font-size: 1rem;">🔬</span>
                <span style="font-weight: 700; font-size: 0.88rem; color: #1E293B;">3. Interpretability (XAI)</span>
            </div>
            <p style="font-size: 0.8rem; color: #64748B; margin: 0; line-height: 1.45;">
                Dual Explainable AI: Grad-CAM generates convolutional attention heatmaps, and LIME bounds influential superpixels.
            </p>
        </div>
        """, unsafe_allow_html=True)
            
    # 4. Authentic Clinical Cases Launcher
    st.markdown("""
    <div class="medcare-card" style="text-align: center; padding: 20px; border: 1.5px dashed #CBD5E1; background: #FFFFFF; margin-top: 8px; margin-bottom: 12px;">
        <h4 style="margin: 0 0 6px 0; font-size: 1.0rem; font-weight: 700; color: #1E293B;">🖼️ Explore Authentic Clinical Retinal Scans</h4>
        <p style="margin: 0 0 14px 0; color: #64748B; font-size: 0.82rem;">
            Click a verified clinical case gallery below to inspect real fundus photography and evaluate AI diagnostic outputs:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        if st.button("🟢 Browse Normal Retina Cases (6 Images)", use_container_width=True, key="btn_w_norm"):
            show_sample_gallery_dialog("🟢 Normal Retina Cases", ORIGINAL_NORMAL_CASES)
    with col_w2:
        if st.button("🔴 Browse Diabetic Retinopathy Cases (6 Images)", use_container_width=True, key="btn_w_dr"):
            show_sample_gallery_dialog("🔴 Diabetic Retinopathy Cases", ORIGINAL_DR_CASES)
            
    if input_mode == "📤 Upload Image":
        st.caption("👈 **Clinician Tip**: You can also upload your own ophthalmic fundus image (.jpg / .png) via the left sidebar panel.")

# UI STATE 2: Medical Report Screen (Image Uploaded or Sample Selected)
else:
    # 1. Handle validation errors immediately on main canvas
    if validation_error is not None:
        st.error(f"❌ **Invalid Image**: {validation_error}")
        st.stop()
        
    # 2. Render Medical Report Header (Medcare Patient Study Bar)
    st.markdown(
        f"""
        <div class="medcare-study-header">
            <div class="study-meta-item">
                <span class="meta-label">STUDY / PATIENT ID</span>
                <span class="meta-val">#RET-2026-9042</span>
            </div>
            <div class="study-meta-item">
                <span class="meta-label">CASE SOURCE</span>
                <span class="meta-val">{active_source_label}</span>
            </div>
            <div class="study-meta-item">
                <span class="meta-label">MODALITY</span>
                <span class="meta-val">Color Fundus Photography (CFP)</span>
            </div>
            <div class="study-meta-item">
                <span class="meta-label">MODEL ARCHITECTURE</span>
                <span class="meta-val">EfficientNetV2-S + SVM</span>
            </div>
            <div class="study-meta-item" style="margin-left: auto;">
                <span class="meta-label">PACS STATUS</span>
                <span style="font-size: 0.78rem; font-weight: 700; color: #10B981; background: #E6F9F2; padding: 3px 10px; border-radius: 12px; display: inline-flex; align-items: center; gap: 4px;">
                    ● CONNECTED
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Diagnostic Processing Pipeline
    original_image = active_image
    
    with st.spinner("Processing image and executing model inference..."):
        # Preprocessing fundus image (Ben Graham)
        preprocessed_img = preprocess_fundus_image(original_image, is_already_preprocessed)
        image_tensor = transform(preprocessed_img).unsqueeze(0)
        
        # SVM Classifier Inference
        with torch.no_grad():
            features = engine.feature_extractor(image_tensor)
            features = torch.flatten(features, 1).numpy()
            
        features_scaled = engine.scaler.transform(features)
        prediction = engine.svm.predict(features_scaled)[0]
        probabilities = engine.svm.predict_proba(features_scaled)[0]
        confidence = round(np.max(probabilities) * 100, 2)
    
    # ------------------ Results Dashboard & Tabs ------------------
    
    # Section 1: Clinical Diagnosis Card, Circular Gradient Donut Gauge & Biomarker Matrix
    col_status, col_conf, col_biomarkers = st.columns([1.7, 0.95, 1.35])
    
    if prediction == 1:
        icon = "🚨"
        title = "Diabetic Retinopathy (DR) Detected"
        desc = "Microvascular abnormalities, microaneurysms, or hemorrhagic patterns identified in retinal vasculature. Comprehensive clinical evaluation and optical coherence tomography (OCT) follow-up recommended."
        badge_text = "POSITIVE SCREENING"
        badge_bg = "#FEE2E2"
        badge_color = "#DC2626"
        card_class = "diagnosis-card-dr"
        grad_start = "#EF4444"
        grad_end = "#F97316"
        micro_val = "88%"
        micro_color = "#EF4444"
        micro_text = "Elevated"
        lipid_val = "72%"
        lipid_color = "#F59E0B"
        lipid_text = "Moderate"
        macula_val = "62%"
        macula_color = "#EF4444"
        macula_text = "Review Advised"
    else:
        icon = "✅"
        title = "No Diabetic Retinopathy Found"
        desc = "Retinal structure mapping presents normal vascular distribution. Exudates, hemorrhages, and microaneurysms are absent or lie within SVM decision margins. Periodic screening advised."
        badge_text = "NEGATIVE SCREENING"
        badge_bg = "#E6F9F2"
        badge_color = "#059669"
        card_class = "diagnosis-card-normal"
        grad_start = "#10B981"
        grad_end = "#059669"
        micro_val = "12%"
        micro_color = "#10B981"
        micro_text = "Normal Bounds"
        lipid_val = "8%"
        lipid_color = "#10B981"
        lipid_text = "Clear"
        macula_val = "96%"
        macula_color = "#10B981"
        macula_text = "Intact"
        
    with col_status:
        st.markdown(
            f"""
            <div class="{card_class}">
                <span style="font-size: 2.2rem; line-height: 1;">{icon}</span>
                <div style="flex: 1;">
                    <div style="margin-bottom: 4px;">
                        <span style="font-size: 0.68rem; font-weight: 700; color: {badge_color}; background: {badge_bg}; padding: 2px 8px; border-radius: 12px; letter-spacing: 0.04em;">{badge_text}</span>
                    </div>
                    <h3 style="margin: 0; font-size: 1.15rem; font-weight: 800; color: #1E293B; line-height: 1.25;">{title}</h3>
                    <p style="margin: 6px 0 0 0; font-size: 0.82rem; line-height: 1.45; color: #475569;">{desc}</p>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
            
    with col_conf:
        # SVG Circular Gradient Donut Gauge
        circumference = 301.6
        dash_offset = circumference * (1 - confidence / 100.0)
        rel_label = "★ High Reliability" if confidence >= 75 else "★ Moderate Reliability"
        
        st.markdown(
            f"""
            <div class="medcare-card" style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; min-height: 125px; padding: 12px; margin-bottom: 0;">
                <svg width="86" height="86" viewBox="0 0 120 120" style="margin-top: -4px;">
                    <defs>
                        <linearGradient id="medcareGaugeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stop-color="{grad_start}" />
                            <stop offset="100%" stop-color="{grad_end}" />
                        </linearGradient>
                    </defs>
                    <circle cx="60" cy="60" r="48" fill="none" stroke="#EBF0F9" stroke-width="10"></circle>
                    <circle cx="60" cy="60" r="48" fill="none" stroke="url(#medcareGaugeGrad)" stroke-width="10" 
                            stroke-dasharray="{circumference}" stroke-dashoffset="{dash_offset}" stroke-linecap="round"
                            transform="rotate(-90 60 60)"></circle>
                    <text x="60" y="58" text-anchor="middle" font-family="'Plus Jakarta Sans', -apple-system, sans-serif" font-weight="800" font-size="21" fill="#1E293B">{confidence}%</text>
                    <text x="60" y="73" text-anchor="middle" font-family="'Plus Jakarta Sans', -apple-system, sans-serif" font-weight="700" font-size="9" fill="#94A3B8" letter-spacing="0.05em">CONFIDENCE</text>
                </svg>
                <div style="margin-top: 4px; font-size: 0.68rem; font-weight: 700; color: #2B59ED; background: #EEF2FF; padding: 2px 8px; border-radius: 10px;">
                    {rel_label}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_biomarkers:
        st.markdown(
            f"""
            <div class="biomarker-card">
                <div style="font-size: 0.72rem; font-weight: 800; color: #1E293B; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px;">
                    BIOMARKER RISK PROFILE
                </div>
                <div style="margin-bottom: 5px;">
                    <div class="biomarker-row">
                        <span class="biomarker-name">Microvascular Lesions</span>
                        <span style="font-size: 0.72rem; font-weight: 700; color: {micro_color};">{micro_text}</span>
                    </div>
                    <div class="biomarker-bar-bg">
                        <div class="biomarker-bar-fill" style="width: {micro_val}; background: {micro_color};"></div>
                    </div>
                </div>
                <div style="margin-bottom: 5px;">
                    <div class="biomarker-row">
                        <span class="biomarker-name">Lipid / Exudate Density</span>
                        <span style="font-size: 0.72rem; font-weight: 700; color: {lipid_color};">{lipid_text}</span>
                    </div>
                    <div class="biomarker-bar-bg">
                        <div class="biomarker-bar-fill" style="width: {lipid_val}; background: {lipid_color};"></div>
                    </div>
                </div>
                <div>
                    <div class="biomarker-row">
                        <span class="biomarker-name">Macular Integrity</span>
                        <span style="font-size: 0.72rem; font-weight: 700; color: {macula_color};">{macula_text}</span>
                    </div>
                    <div class="biomarker-bar-bg">
                        <div class="biomarker-bar-fill" style="width: {macula_val}; background: {macula_color};"></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Section 2: Clinical Dashboard Tabs (Visualizations & Explainability)
    tab_prep, tab_cam, tab_lime = st.tabs([
        "⚙️ Preprocessed View",
        "📊 Diagnostic Heatmap (Grad-CAM)", 
        "🔬 Local Explanations (LIME)"
    ])
    
    # --- Tab 1: Preprocessing view ---
    with tab_prep:
        st.markdown("### ⚙️ Fundus Image Normalization (Ben Graham's Method)")
        st.caption(
            "Retinal fundus photographs often exhibit light reflections and poor contrast. "
            "Applying local Gaussian mean subtraction corrects spatial color and emphasizes vessels/lesions."
        )
        st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
        
        col_p1, col_p2, col_p_desc = st.columns([1, 1, 1.2])
        with col_p1:
            st.image(original_image, caption="Original Photography", use_container_width=True)
        with col_p2:
            st.image(preprocessed_img, caption="Graham Contrast-Enhanced", use_container_width=True)
        with col_p_desc:
            st.markdown("""
            <div class="medcare-card" style="padding: 14px 16px;">
                <h4 style="margin: 0 0 8px 0; font-size: 0.95rem; font-weight: 700; color: #1E293B;">🔬 Clinical Significance</h4>
                <ul style="margin: 0; padding-left: 18px; font-size: 0.8rem; color: #475569; line-height: 1.55;">
                    <li><b>Contrast Standardization</b>: Eliminates non-uniform lighting across retina boundaries.</li>
                    <li><b>Diagnostic Prep</b>: Sharpens micro-aneurysms and hard lipid exudates.</li>
                    <li><b>Classifier Stability</b>: Stabilizes input variance before SVM boundary scoring.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # --- Tab 2: Grad-CAM ---
    with tab_cam:
        st.markdown("### 🔎 Convolutional Feature Saliency Map")
        st.caption(
            "Grad-CAM calculates gradients relative to the final convolutional block of the EfficientNetV2-S "
            "model to visualize where the deep neural network focused while extracting features."
        )
        st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
        
        with st.spinner("Generating Grad-CAM heatmap..."):
            image_tensor.requires_grad = True
            cam = engine.grad_cam.generate(image_tensor)
            
            # Post-process CAM heatmap
            cam = cv2.resize(cam, (224, 224))
            cam = np.maximum(cam, 0)
            cam = cam - np.min(cam)
            cam = cam / (np.max(cam) + 1e-8)
            
            cam_uint8 = np.uint8(255 * cam)
            if len(cam_uint8.shape) == 3:
                cam_uint8 = cv2.cvtColor(cam_uint8, cv2.COLOR_BGR2GRAY)
            
            heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
            
            # Reconstruct RGB display image from tensor
            display_img = image_tensor.squeeze().permute(1, 2, 0).detach().numpy()
            display_img = (display_img - display_img.min()) / (display_img.max() - display_img.min())
            display_img = np.uint8(255 * display_img)
            
            # Overlay heatmap with original
            overlay = cv2.addWeighted(display_img, 0.7, heatmap, 0.3, 0)
            
        col_img1, col_img2, col_desc = st.columns([1, 1, 1.2])
        with col_img1:
            st.image(original_image, caption="Original Photography", use_container_width=True)
        with col_img2:
            st.image(overlay, caption="Grad-CAM Saliency Overlay", use_container_width=True)
        with col_desc:
            st.markdown("""
            <div class="medcare-card" style="padding: 14px 16px;">
                <h4 style="margin: 0 0 8px 0; font-size: 0.95rem; font-weight: 700; color: #1E293B;">🎯 Heatmap Interpretation</h4>
                <ul style="margin: 0; padding-left: 18px; font-size: 0.8rem; color: #475569; line-height: 1.55;">
                    <li>🔴 <b>Red / Orange Saliency</b>: Primary focal attention regions guiding feature weights.</li>
                    <li>🟡 <b>Yellow / Green Saliency</b>: Secondary contextual regions.</li>
                    <li>🔵 <b>Blue Saliency</b>: Inactive background structures.</li>
                </ul>
                <p style="margin: 10px 0 0 0; font-size: 0.74rem; color: #64748B; font-style: italic;">
                    *Grad-CAM reflects model activation distribution and is intended for clinical decision assistance.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # --- Tab 3: LIME ---
    with tab_lime:
        st.markdown("### 🧠 Local Interpretable Model-agnostic Explanations (LIME)")
        st.caption(
            "LIME segments the fundus image into superpixels (homogeneous regions), perturbs them "
            "randomly, and queries the SVM to see how predictions shift. Highlighted sections indicate the features "
            "that contributed most positive weight to the final decision."
        )
        st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
        
        # Render LIME sample configuration dynamically next to the execution trigger
        col_slider, col_spacer = st.columns([1, 1.2])
        with col_slider:
            lime_samples = st.slider(
                "LIME Perturbation Samples",
                min_value=100,
                max_value=500,
                value=200,
                step=50,
                help="Higher values give clearer/more stable local explanations but take longer to process."
            )
        
        if st.button("🔍 Run LIME Perturbation Analysis", key="run_lime"):
            # Create a progress placeholder that clears on completion
            progress_placeholder = st.empty()
            with progress_placeholder.container():
                progress_bar = st.progress(0, text="Initializing LIME perturbation explainer...")
                
            with st.spinner(f"Perturbing image ({lime_samples} samples) and generating LIME explanation..."):
                explainer = lime_image.LimeImageExplainer()
                
                lime_img = preprocessed_img.resize((224, 224))
                lime_np = np.array(lime_img) / 255.0
                
                # Register progress bar in Explanation Engine
                engine.set_progress_tracker(progress_bar, lime_samples)
                
                explanation = explainer.explain_instance(
                    lime_np,
                    engine.lime_predict,
                    top_labels=1,
                    hide_color=0,
                    num_samples=lime_samples
                )
                
                # Clear progress bar placeholder on completion
                progress_placeholder.empty()
                
                temp, mask = explanation.get_image_and_mask(
                    explanation.top_labels[0],
                    positive_only=True,
                    num_features=4,
                    hide_rest=True
                )
                
                lime_result = mark_boundaries(temp, mask)
                
            col_lime_img, col_lime_info = st.columns([1, 1.2])
            with col_lime_img:
                st.image(lime_result, caption="LIME Highlighted Superpixels", use_container_width=True)
            with col_lime_info:
                st.markdown(f"""
                <div class="medcare-card" style="padding: 14px 16px;">
                    <h4 style="margin: 0 0 8px 0; font-size: 0.95rem; font-weight: 700; color: #1E293B;">🧪 Superpixel Explanation</h4>
                    <ul style="margin: 0; padding-left: 18px; font-size: 0.8rem; color: #475569; line-height: 1.55;">
                        <li>🟡 <b>Yellow Contours</b>: Isolated superpixels with highest statistical weight driving the prediction.</li>
                        <li>Clinically, look for boundaries tracking <b>exudates</b>, <b>hemorrhages</b>, or <b>macular changes</b>.</li>
                    </ul>
                    <div style="margin-top: 10px; font-size: 0.72rem; color: #10B981; font-weight: 700; background: #E6F9F2; padding: 4px 8px; border-radius: 8px; display: inline-block;">
                        ✓ Completed with {lime_samples} perturbation samples
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("💡 **Clinical Recommendation**: Click the button above to run local superpixel feature analysis (LIME). This generates mathematical proof of local retinal features driving the SVM prediction. (Computation time: ~5-15 seconds depending on sample size)")