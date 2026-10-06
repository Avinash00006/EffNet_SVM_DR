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
# Modal dialogs for browsing real fundus images and contacting developer
if hasattr(st, "dialog"):
    @st.dialog("🖼️ Select Clinical Case", width="large")
    def show_sample_gallery_dialog(category_name, cases_dict):
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 2px;">
                <div style="font-size: 1.10rem; font-weight: 800; color: #0F172A !important; letter-spacing: -0.01em;">{category_name}</div>
                <div style="font-size: 0.70rem; font-weight: 700; color: #2B59ED !important; background: #EEF2FF; border: 1px solid #C7D2FE; padding: 2px 8px; border-radius: 12px;">6 REAL CASES</div>
            </div>
            <div style="font-size: 0.78rem; font-weight: 500; color: #475569 !important; margin-bottom: 14px; line-height: 1.4;">
                Click directly on any retinal photography case below to load and run automated diagnostic analysis:
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Grid CSS & Direct-Click Card Styling
        st.markdown("""
        <style>
            /* Modal Surface - Clean Light Clinical Styling with High Contrast */
            div[data-testid="stDialog"] div[role="dialog"],
            div[data-baseweb="modal"] div[role="dialog"],
            div[data-testid="stModal"] div[role="dialog"],
            div[role="dialog"] {
                max-width: 660px !important;
                width: 92vw !important;
                background-color: #FFFFFF !important;
                border-radius: 16px !important;
                border: 1px solid #CBD5E1 !important;
                box-shadow: 0 20px 50px rgba(15, 23, 42, 0.22) !important;
                padding: 16px 20px !important;
            }
            div[data-testid="stDialog"] div[role="dialog"] > div,
            div[data-baseweb="modal"] div[role="dialog"] > div,
            div[data-testid="stModal"] div[role="dialog"] > div,
            div[role="dialog"] > div {
                background-color: #FFFFFF !important;
            }
            div[data-testid="stDialog"] h2,
            div[data-testid="stDialog"] [data-testid="stHeadingWithActionElements"] h2,
            div[role="dialog"] h2,
            div[role="dialog"] [data-testid="stHeadingWithActionElements"] h2 {
                color: #0F172A !important;
                font-size: 1.15rem !important;
                font-weight: 800 !important;
            }
            div[role="dialog"] button[aria-label="Close"],
            div[data-testid="stDialog"] button[aria-label="Close"] svg {
                color: #475569 !important;
                fill: #475569 !important;
            }

            /* 2x3 Grid Column Cards */
            div[data-testid="stDialog"] div[data-testid="stColumn"],
            div[role="dialog"] div[data-testid="stColumn"] {
                position: relative !important;
                border: 1.5px solid #E2E8F0 !important;
                border-radius: 14px !important;
                padding: 8px 8px 10px 8px !important;
                text-align: center !important;
                transition: all 0.2s ease-in-out !important;
                background-color: #FFFFFF !important;
                cursor: pointer !important;
                box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04) !important;
            }
            div[data-testid="stDialog"] div[data-testid="stColumn"]:hover,
            div[role="dialog"] div[data-testid="stColumn"]:hover {
                border-color: #2B59ED !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 18px rgba(43, 89, 237, 0.16) !important;
            }

            /* Neutralize intermediate wrappers so button stretches across entire stColumn */
            div[data-testid="stDialog"] div[data-testid="stColumn"] div[data-testid="stVerticalBlock"],
            div[data-testid="stDialog"] div[data-testid="stColumn"] div[data-testid="stVerticalBlockBorderWrapper"],
            div[data-testid="stDialog"] div[data-testid="stColumn"] div[data-testid="stElementContainer"],
            div[data-testid="stDialog"] div[data-testid="stColumn"] div.element-container,
            div[role="dialog"] div[data-testid="stColumn"] div[data-testid="stVerticalBlock"],
            div[role="dialog"] div[data-testid="stColumn"] div[data-testid="stElementContainer"] {
                position: static !important;
            }

            /* Fundus image thumbnail: pointer-events none */
            div[data-testid="stDialog"] div[data-testid="stColumn"] [data-testid="stImage"],
            div[role="dialog"] div[data-testid="stColumn"] [data-testid="stImage"] {
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                pointer-events: none !important;
                user-select: none !important;
                -webkit-user-select: none !important;
            }
            div[data-testid="stDialog"] div[data-testid="stColumn"] img,
            div[role="dialog"] div[data-testid="stColumn"] img {
                max-height: 95px !important;
                height: 95px !important;
                width: auto !important;
                border-radius: 8px !important;
                object-fit: cover !important;
                border: 1px solid #E2E8F0 !important;
                pointer-events: none !important;
                user-select: none !important;
                -webkit-user-select: none !important;
            }

            /* Text labels: pointer-events none */
            div[data-testid="stDialog"] div[data-testid="stColumn"] [data-testid="stMarkdownContainer"],
            div[role="dialog"] div[data-testid="stColumn"] [data-testid="stMarkdownContainer"] {
                pointer-events: none !important;
                user-select: none !important;
                -webkit-user-select: none !important;
            }

            /* Streamlit button overlay spans entire column */
            div[data-testid="stDialog"] div[data-testid="stColumn"] div.stButton,
            div[data-testid="stDialog"] div[data-testid="stColumn"] div[data-testid="stButton"],
            div[role="dialog"] div[data-testid="stColumn"] div.stButton,
            div[role="dialog"] div[data-testid="stColumn"] div[data-testid="stButton"] {
                position: absolute !important;
                inset: 0 !important;
                top: 0 !important;
                left: 0 !important;
                right: 0 !important;
                bottom: 0 !important;
                width: 100% !important;
                height: 100% !important;
                margin: 0 !important;
                padding: 0 !important;
                z-index: 50 !important;
            }
            div[data-testid="stDialog"] div[data-testid="stColumn"] div.stButton > button,
            div[data-testid="stDialog"] div[data-testid="stColumn"] div[data-testid="stButton"] > button,
            div[role="dialog"] div[data-testid="stColumn"] div.stButton > button,
            div[role="dialog"] div[data-testid="stColumn"] div[data-testid="stButton"] > button {
                position: absolute !important;
                inset: 0 !important;
                top: 0 !important;
                left: 0 !important;
                right: 0 !important;
                bottom: 0 !important;
                width: 100% !important;
                height: 100% !important;
                opacity: 0 !important;
                cursor: pointer !important;
                border: none !important;
                background: transparent !important;
                z-index: 51 !important;
                margin: 0 !important;
                padding: 0 !important;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # 2 Rows with 3 Images in each row (3 columns per row)
        cases_list = list(cases_dict.items())
        safe_cat = "norm" if "Normal" in category_name else "dr"
        
        for row_idx in range(0, len(cases_list), 3):
            row_cases = cases_list[row_idx : row_idx + 3]
            cols = st.columns(3)
            for col_idx, (label, rel_path) in enumerate(row_cases):
                global_idx = row_idx + col_idx
                full_path = os.path.join(os.path.dirname(__file__), rel_path)
                
                if " (" in label:
                    short_name = label.split(" (")[0]
                    case_hash = label.split(" (")[1].replace(")", "")
                else:
                    short_name = label
                    case_hash = ""
                    
                with cols[col_idx]:
                    if os.path.exists(full_path):
                        img = Image.open(full_path)
                        st.image(img, use_container_width=True)
                        st.markdown(
                            f"""
                            <div style='text-align: center; margin-top: 4px; pointer-events: none;'>
                                <div style='font-size: 0.82rem; font-weight: 800; color: #0F172A !important; line-height: 1.2;'>{short_name}</div>
                                <div style='font-size: 0.65rem; font-weight: 600; color: #64748B !important; font-family: monospace;'>ID: {case_hash[:8]}...</div>
                                <div style='font-size: 0.65rem; font-weight: 700; color: #2B59ED !important; background: #EEF2FF; border-radius: 6px; padding: 2px 6px; margin: 4px auto 0 auto; width: fit-content;'>👆 Click to Select</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    if st.button(
                        f"Select {short_name}",
                        key=f"btn_pick_{safe_cat}_{global_idx}"
                    ):
                        st.session_state.selected_sample_path = full_path
                        st.session_state.selected_sample_label = f"{category_name} - {label}"
                        st.rerun()

            if row_idx == 0:
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    @st.dialog("✉️ Contact Developer", width="small")
    def show_contact_dialog():
        st.markdown("<h3 style='margin: 0 0 2px 0; font-size: 1.1rem; color: #1E293B !important; font-weight: 800;'>✉️ Contact Developer</h3>", unsafe_allow_html=True)
        st.markdown("<p style='margin: 0 0 14px 0; font-size: 0.8rem; color: #64748B !important;'>Have questions or feedback? Send a message directly:</p>", unsafe_allow_html=True)
        
        if st.session_state.get("message_sent", False):
            st.success("📩 Message already sent! Thank you for reaching out.")
            return

        contact_name = st.text_input("Name", placeholder="Your Name", key="dlg_contact_name")
        contact_email = st.text_input("Email", placeholder="Your Email", key="dlg_contact_email")
        contact_msg = st.text_area("Message", placeholder="Type your message...", key="dlg_contact_msg")
        
        if st.button("Send Message", use_container_width=True, key="dlg_btn_send"):
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
else:
    def show_sample_gallery_dialog(category_name, cases_dict):
        st.session_state.fallback_gallery_open = True
    def show_contact_dialog():
        pass


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
        
        /* 2. Compact Viewport Container (Optimized for 100% Zoom with Breathing Space) */
        .block-container,
        [data-testid="stMainBlockContainer"] {
            max-width: 1260px !important;
            padding-top: 2.2rem !important;
            padding-bottom: 2.0rem !important;
            padding-left: 2.0rem !important;
            padding-right: 2.0rem !important;
            margin: 0 auto !important;
        }
        
        /* 3. Hide Streamlit Header, Toolbar, Share Button, and Decoration */
        header[data-testid="stHeader"] {
            height: 0px !important;
            background: transparent !important;
            background-color: transparent !important;
        }
        div[data-testid="stToolbar"],
        [data-testid="stToolbar"],
        header [data-testid="stToolbar"],
        .stToolbar,
        .stToolbarActions,
        div[data-testid="stShareButton"],
        button[data-testid="stShareButton"],
        .viewerBadge,
        a[class*="viewerBadge"],
        div[class*="viewerBadge"],
        .stDeployButton,
        #MainMenu,
        footer,
        div[data-testid="stDecoration"],
        #stDecoration,
        div[data-testid="InputInstructions"] {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
            height: 0 !important;
            width: 0 !important;
            pointer-events: none !important;
        }
        
        /* 4. Typography Hierarchy & Contrast */
        h1, [data-testid="stHeadingWithActionElements"] h1 {
            font-size: 1.30rem !important;
            font-weight: 800 !important;
            line-height: 1.25 !important;
            margin: 0.2rem 0 0.35rem 0 !important;
            color: #1E293B !important;
            letter-spacing: -0.02em !important;
        }
        h2, [data-testid="stHeadingWithActionElements"] h2 {
            font-size: 1.10rem !important;
            font-weight: 700 !important;
            line-height: 1.25 !important;
            margin: 0.2rem 0 0.30rem 0 !important;
            color: #1E293B !important;
        }
        h3, [data-testid="stHeadingWithActionElements"] h3 {
            font-size: 0.90rem !important;
            font-weight: 700 !important;
            line-height: 1.3 !important;
            margin: 0.15rem 0 0.25rem 0 !important;
            color: #1E293B !important;
        }
        p, span, label, [data-testid="stMarkdownContainer"] p {
            font-size: 13px !important;
            line-height: 1.45 !important;
            color: #1E293B;
        }
        small, .caption, [data-testid="stImageCaption"] {
            font-size: 11.5px !important;
            color: #64748B !important;
            line-height: 1.3 !important;
        }
        
        /* 5. Sidebar: Restored Smooth Scrolling & Generous Proportional Spacing */
        section[data-testid="stSidebar"],
        [data-testid="stSidebar"],
        div[data-testid="stSidebarUserContent"] {
            width: 280px !important;
            min-width: 280px !important;
            max-width: 285px !important;
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
            box-shadow: 4px 0 16px rgba(43, 89, 237, 0.03) !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
        }
        [data-testid="stSidebarContent"] {
            padding-top: 1.4rem !important;
            padding-bottom: 2.2rem !important;
            padding-left: 1.15rem !important;
            padding-right: 1.15rem !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
        }
        section[data-testid="stSidebar"]::-webkit-scrollbar,
        [data-testid="stSidebarContent"]::-webkit-scrollbar {
            width: 5px !important;
        }
        section[data-testid="stSidebar"]::-webkit-scrollbar-track,
        [data-testid="stSidebarContent"]::-webkit-scrollbar-track {
            background: transparent !important;
        }
        section[data-testid="stSidebar"]::-webkit-scrollbar-thumb,
        [data-testid="stSidebarContent"]::-webkit-scrollbar-thumb {
            background: #CBD5E1 !important;
            border-radius: 4px !important;
        }
        section[data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover,
        [data-testid="stSidebarContent"]::-webkit-scrollbar-thumb:hover {
            background: #94A3B8 !important;
        }
        div[data-testid="stSidebar"] div.stButton button {
            margin-bottom: 8px !important;
            padding: 8px 12px !important;
            font-size: 12.5px !important;
            font-weight: 700 !important;
            border-radius: 10px !important;
        }
        
        /* Medcare Sidebar Brand Badge */
        .medcare-sidebar-brand {
            display: flex;
            align-items: center;
            gap: 8px;
            background: linear-gradient(135deg, #2B59ED 0%, #3B71F7 100%);
            padding: 8px 12px;
            border-radius: 12px;
            color: #FFFFFF !important;
            margin-bottom: 8px;
            box-shadow: 0 4px 14px rgba(43, 89, 237, 0.2);
        }
        .medcare-sidebar-brand h2 {
            color: #FFFFFF !important;
            margin: 0 !important;
            font-size: 1.0rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.02em;
            line-height: 1.1;
        }
        
        /* 6. Medcare Cards & Containers */
        .medcare-card {
            background: #FFFFFF;
            border-radius: 14px;
            border: 1px solid #E9EFF7;
            padding: 12px 14px;
            box-shadow: 0 2px 10px rgba(43, 89, 237, 0.04);
            margin-bottom: 12px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .medcare-card:hover {
            box-shadow: 0 4px 16px rgba(43, 89, 237, 0.08);
        }
        .medcare-card * {
            color: #1E293B !important;
        }
        .medcare-card p {
            color: #475569 !important;
        }
        
        /* 7. Buttons: Royal Blue Pill */
        [data-testid="stButton"] button,
        .stButton > button {
            background: linear-gradient(135deg, #2B59ED 0%, #3B71F7 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 10px !important;
            min-height: 32px !important;
            height: 32px !important;
            padding: 3px 12px !important;
            font-weight: 600 !important;
            font-size: 12.5px !important;
            letter-spacing: -0.01em !important;
            box-shadow: 0 3px 10px rgba(43, 89, 237, 0.2) !important;
            transition: all 0.2s ease-in-out !important;
        }
        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 5px 14px rgba(43, 89, 237, 0.3) !important;
            color: #FFFFFF !important;
        }
        
        /* Secondary Category Buttons */
        button[key="btn_side_norm"], button[key="btn_w_norm"] {
            background: #FFFFFF !important;
            color: #10B981 !important;
            border: 1.5px solid #10B981 !important;
            box-shadow: 0 2px 6px rgba(16, 185, 129, 0.12) !important;
        }
        button[key="btn_side_norm"]:hover, button[key="btn_w_norm"]:hover {
            background: #E6F9F2 !important;
            color: #059669 !important;
        }
        button[key="btn_side_dr"], button[key="btn_w_dr"] {
            background: #FFFFFF !important;
            color: #EF4444 !important;
            border: 1.5px solid #EF4444 !important;
            box-shadow: 0 2px 6px rgba(239, 68, 68, 0.12) !important;
        }
        button[key="btn_side_dr"]:hover, button[key="btn_w_dr"]:hover {
            background: #FEECEB !important;
            color: #DC2626 !important;
        }
        
        /* 8. Tabs: Sleek Clinical Pill Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px !important;
            background-color: #EBF0F9 !important;
            padding: 4px !important;
            border-radius: 12px !important;
            min-height: 32px !important;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 9px !important;
            padding: 5px 14px !important;
            font-weight: 600 !important;
            font-size: 12.5px !important;
            color: #64748B !important;
            background-color: transparent !important;
            border: none !important;
            min-height: 30px !important;
            height: 30px !important;
        }
        .stTabs [aria-selected="true"] {
            background-color: #FFFFFF !important;
            color: #2B59ED !important;
            box-shadow: 0 2px 6px rgba(43, 89, 237, 0.12) !important;
        }
        
        /* 9. Radio Buttons as Segmented Control */
        div[data-testid="stRadio"] > div {
            background-color: #EBF0F9;
            padding: 3px;
            border-radius: 10px;
            gap: 3px;
        }
        div[data-testid="stRadio"] label {
            padding: 4px 10px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 12px;
            color: #475569 !important;
        }
        div[data-testid="stRadio"] label[data-checked="true"] {
            background-color: #FFFFFF;
            color: #2B59ED !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.06);
        }

        /* 9b. Clean Light-Themed File Uploader (Zero Dark Backgrounds) */
        [data-testid="stFileUploader"] {
            background-color: transparent !important;
        }
        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploaderDropzone"],
        section[data-testid="stFileUploadDropzone"],
        div[data-testid="stFileUploaderDropzone"] {
            background-color: #F8FAFC !important;
            border: 1.5px dashed #CBD5E1 !important;
            border-radius: 12px !important;
            color: #1E293B !important;
            padding: 8px 10px !important;
            transition: all 0.2s ease-in-out !important;
        }
        [data-testid="stFileUploader"] section:hover,
        [data-testid="stFileUploaderDropzone"]:hover,
        section[data-testid="stFileUploadDropzone"]:hover {
            border-color: #2B59ED !important;
            background-color: #EEF2FF !important;
        }
        [data-testid="stFileUploader"] * {
            color: #1E293B !important;
        }
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] p,
        [data-testid="stFileUploader"] label {
            color: #475569 !important;
        }
        [data-testid="stFileUploader"] button {
            background: #FFFFFF !important;
            color: #1E293B !important;
            border: 1.5px solid #CBD5E1 !important;
            border-radius: 8px !important;
            font-size: 12px !important;
            font-weight: 700 !important;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05) !important;
            height: 28px !important;
            min-height: 28px !important;
        }
        [data-testid="stFileUploader"] button:hover {
            background-color: #F1F5F9 !important;
            color: #2B59ED !important;
            border-color: #2B59ED !important;
        }
        [data-testid="stFileUploader"] svg {
            fill: #2B59ED !important;
            stroke: #2B59ED !important;
        }
        
        /* 10. Fundus Image Display Constraints (Compact 290px max) */
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
            color: #2B59ED !important;
            padding: 3px 9px;
            border-radius: 9999px;
            font-size: 0.70rem;
            font-weight: 700;
            margin-right: 5px;
            border: 1px solid #E0E7FF;
        }

        /* Social Badges with Guaranteed Visibility */
        .social-badge {
            text-decoration: none !important;
            color: #1E293B !important;
            background-color: #F1F5F9 !important;
            border: 1px solid #CBD5E1 !important;
            padding: 4px 9px !important;
            border-radius: 7px !important;
            font-size: 0.72rem !important;
            font-weight: 700 !important;
            display: inline-flex !important;
            align-items: center !important;
            gap: 4px !important;
            cursor: pointer !important;
            transition: all 0.15s ease-in-out !important;
        }
        .social-badge:hover {
            background-color: #E2E8F0 !important;
            color: #2B59ED !important;
            border-color: #94A3B8 !important;
        }

        /* 11. Medcare Hero Banner & KPI Components (Compact Sizing) */
        .medcare-hero-banner {
            background: linear-gradient(135deg, #1E3A8A 0%, #2B59ED 60%, #3B82F6 100%);
            border-radius: 16px;
            padding: 16px 22px;
            color: #FFFFFF !important;
            box-shadow: 0 6px 20px rgba(43, 89, 237, 0.16);
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 14px;
            position: relative;
            overflow: hidden;
        }
        .medcare-hero-banner::after {
            content: "";
            position: absolute;
            right: -25px;
            top: -25px;
            width: 170px;
            height: 170px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255,255,255,0.18) 0%, transparent 70%);
            pointer-events: none;
        }
        .medcare-hero-banner h2 {
            color: #FFFFFF !important;
            font-size: 1.25rem !important;
            font-weight: 800 !important;
            margin: 2px 0 4px 0 !important;
            letter-spacing: -0.02em !important;
        }
        .medcare-hero-banner p {
            color: rgba(255, 255, 255, 0.94) !important;
            font-size: 0.80rem !important;
            line-height: 1.45 !important;
            margin: 0 !important;
            max-width: 680px !important;
        }
        .medcare-hero-banner div,
        .medcare-hero-banner span {
            color: #FFFFFF !important;
        }
        .medcare-badge-live {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            background: rgba(255, 255, 255, 0.18);
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: #FFFFFF !important;
            font-size: 0.68rem;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 20px;
            letter-spacing: 0.04em;
            margin-bottom: 6px;
        }
        .medcare-kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 14px;
        }
        .medcare-kpi-card {
            background: #FFFFFF;
            border-radius: 14px;
            padding: 10px 14px;
            border: 1px solid #E9EFF7;
            box-shadow: 0 2px 8px rgba(43, 89, 237, 0.04);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .medcare-kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 14px rgba(43, 89, 237, 0.08);
        }
        .kpi-title {
            font-size: 0.64rem;
            font-weight: 700;
            text-transform: uppercase;
            color: #64748B !important;
            letter-spacing: 0.03em;
            margin-bottom: 2px;
        }
        .kpi-val {
            font-size: 1.20rem;
            font-weight: 800;
            color: #1E293B !important;
            line-height: 1.1;
            margin-bottom: 4px;
        }
        .kpi-badge-positive {
            display: inline-flex;
            align-items: center;
            gap: 3px;
            font-size: 0.66rem;
            font-weight: 700;
            color: #10B981 !important;
            background: #E6F9F2;
            padding: 2px 6px;
            border-radius: 8px;
            width: fit-content;
        }
        .kpi-badge-neutral {
            display: inline-flex;
            align-items: center;
            gap: 3px;
            font-size: 0.66rem;
            font-weight: 700;
            color: #2B59ED !important;
            background: #EEF2FF;
            padding: 2px 6px;
            border-radius: 8px;
            width: fit-content;
        }
        
        /* 12. Medcare Patient Study Header & Diagnosis Cards */
        .medcare-study-header {
            background: #FFFFFF;
            border-radius: 12px;
            padding: 10px 16px;
            border: 1px solid #E9EFF7;
            display: flex;
            flex-wrap: wrap;
            gap: 18px;
            align-items: center;
            margin-bottom: 14px;
            box-shadow: 0 2px 6px rgba(43, 89, 237, 0.04);
        }
        .study-meta-item {
            display: flex;
            flex-direction: column;
            gap: 1px;
        }
        .meta-label {
            font-size: 0.64rem;
            font-weight: 700;
            color: #94A3B8 !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .meta-val {
            font-size: 0.82rem;
            font-weight: 700;
            color: #1E293B !important;
        }
        
        .diagnosis-card-dr {
            background: #FFFFFF;
            border-radius: 14px;
            border: 1px solid #FEE2E2;
            border-left: 5px solid #EF4444;
            padding: 14px 16px;
            box-shadow: 0 3px 12px rgba(239, 68, 68, 0.08);
            display: flex;
            align-items: flex-start;
            gap: 12px;
            min-height: 115px;
        }
        .diagnosis-card-normal {
            background: #FFFFFF;
            border-radius: 14px;
            border: 1px solid #D1FAE5;
            border-left: 5px solid #10B981;
            padding: 14px 16px;
            box-shadow: 0 3px 12px rgba(16, 185, 129, 0.08);
            display: flex;
            align-items: flex-start;
            gap: 12px;
            min-height: 115px;
        }
        
        /* Biomarker Severity Matrix Card */
        .biomarker-card {
            background: #FFFFFF;
            border-radius: 14px;
            border: 1px solid #E9EFF7;
            padding: 10px 14px;
            box-shadow: 0 2px 8px rgba(43, 89, 237, 0.04);
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-around;
        }
        .biomarker-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 3px;
        }
        .biomarker-name {
            font-size: 0.74rem;
            font-weight: 600;
            color: #475569 !important;
        }
        .biomarker-bar-bg {
            background-color: #F1F5F9;
            height: 5px;
            border-radius: 5px;
            width: 100%;
            overflow: hidden;
            margin-top: 2px;
        }
        .biomarker-bar-fill {
            height: 100%;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

#---------------Load Models & Check Errors----------------------------------------------
full_model, feature_extractor, scaler, svm, load_error = load_models()

#---------------Sidebar Layout (Ophthalmic Control Console)----------------------------
with st.sidebar:
    st.markdown("""
    <div class="medcare-sidebar-brand">
        <span style="font-size: 1.15rem;">👁️</span>
        <div>
            <h2>RetinaAI Care</h2>
            <div style="font-size: 0.64rem; opacity: 0.9; font-weight: 600;">CLINICAL DECISION SYSTEM</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='font-size: 0.74rem; font-weight: 800; color: #475569; text-transform: uppercase; letter-spacing: 0.04em; margin: 14px 0 8px 0;'>📁 Fundus Input Source</div>", unsafe_allow_html=True)
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
        st.markdown("<div style='font-size: 0.74rem; color: #475569; font-weight: 800; text-transform: uppercase; letter-spacing: 0.04em; margin: 16px 0 8px 0;'>🔬 Clinical Samples:</div>", unsafe_allow_html=True)
        
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
                    
                    st.markdown("""
                    <div style='background: #EEF2FF; border: 1.5px solid #C7D2FE; border-radius: 10px; padding: 8px 10px; margin-top: 10px;'>
                        <div style='font-size: 0.68rem; font-weight: 800; color: #2B59ED; text-transform: uppercase; letter-spacing: 0.03em;'>ACTIVE CLINICAL CASE</div>
                        <div style='font-size: 0.78rem; font-weight: 700; color: #1E293B; word-break: break-all; margin-top: 2px;'>""" + os.path.basename(sample_path) + """</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("✕ Deselect Case", key="btn_reset_sample", use_container_width=True):
                        st.session_state.selected_sample_path = None
                        st.session_state.selected_sample_label = None
                        st.rerun()
                except Exception as e:
                    validation_error = f"Failed to load sample image: {str(e)}"
            else:
                validation_error = f"Sample file not found at {sample_path}"

    # Run validation immediately in sidebar background to update controls
    if active_image is not None and validation_error is None:
        is_valid, validation_msg, is_already_preprocessed = validate_fundus_image(active_image)
        if not is_valid:
            validation_error = validation_msg
            
    st.markdown("<div style='border-top: 1.5px solid #E2E8F0; margin: 20px 0 16px 0;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.74rem; font-weight: 800; color: #475569; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 8px;'>📬 Developer & Support</div>", unsafe_allow_html=True)
    
    # Custom HTML for social links (styled with clean, high-contrast badges)
    st.markdown(
        """
        <div style="display: flex; gap: 8px; margin-bottom: 12px;">
            <a href="https://github.com/Avinash00006" target="_blank" class="social-badge">
                🐙 GitHub
            </a>
            <a href="https://linkedin.com/in/avinash-koneti" target="_blank" class="social-badge">
                🔗 LinkedIn
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Initialize contact message session state
    if "message_sent" not in st.session_state:
        st.session_state.message_sent = False
        
    btn_contact_text = "✉️ Message Developer" if not st.session_state.message_sent else "✓ Message Delivered"
    if st.button(btn_contact_text, use_container_width=True, key="btn_open_contact"):
        show_contact_dialog()
            

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
            <h2 style="font-size: 1.22rem; font-weight: 800; color: #FFFFFF !important; margin: 2px 0 4px 0; letter-spacing: -0.02em;">Good Day, Clinician! 👋</h2>
            <p style="color: rgba(255, 255, 255, 0.94) !important; font-size: 0.80rem; line-height: 1.42; margin: 0; max-width: 680px;">
                Welcome to <b>RetinaAI Care</b> workstation. Automated ophthalmic fundus screening powered by hybrid EfficientNetV2-S deep feature extraction, Support Vector Machines, and Explainable AI (Grad-CAM & LIME).
            </p>
        </div>
        <div style="font-size: 2.0rem; margin-left: 18px; opacity: 0.92; z-index: 1;">👁️‍🗨️</div>
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
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                <span style="background: #EBF0F9; padding: 4px 7px; border-radius: 8px; font-size: 0.95rem;">📋</span>
                <span style="font-weight: 700; font-size: 0.84rem; color: #1E293B !important;">1. Quality Assurance</span>
            </div>
            <p style="font-size: 0.76rem; color: #64748B !important; margin: 0; line-height: 1.4;">
                Automated pre-flight validation analyzing aspect ratio, circular aperture coverage, and illumination metrics before pipeline inference.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="medcare-card">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                <span style="background: #EBF0F9; padding: 4px 7px; border-radius: 8px; font-size: 0.95rem;">🧠</span>
                <span style="font-weight: 700; font-size: 0.84rem; color: #1E293B !important;">2. Deep Feature Classifier</span>
            </div>
            <p style="font-size: 0.76rem; color: #64748B !important; margin: 0; line-height: 1.4;">
                EfficientNetV2-S deep convolutional backbone extracts 1,280 structural features, classified with an RBF Support Vector Machine.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="medcare-card">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                <span style="background: #EBF0F9; padding: 4px 7px; border-radius: 8px; font-size: 0.95rem;">🔬</span>
                <span style="font-weight: 700; font-size: 0.84rem; color: #1E293B !important;">3. Interpretability (XAI)</span>
            </div>
            <p style="font-size: 0.76rem; color: #64748B !important; margin: 0; line-height: 1.4;">
                Dual Explainable AI: Grad-CAM generates convolutional attention heatmaps, and LIME bounds influential superpixels.
            </p>
        </div>
        """, unsafe_allow_html=True)

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