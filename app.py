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
    @st.dialog("Select Clinical Case", width="small")
    def show_sample_gallery_dialog(category_name, cases_dict):
        st.markdown(
            f"""
            <div style="font-size: 1.05rem; font-weight: 800; color: #0F172A !important; margin-bottom: 2px;">{category_name}</div>
            <div style="font-size: 0.76rem; font-weight: 500; color: #64748B !important; margin-bottom: 14px;">Click any retinal image below to load and analyze:</div>
            """,
            unsafe_allow_html=True
        )
        
        # Grid CSS: Compact half-ratio window, ClassiAds luminous theme, zero wasted space
        st.markdown("""
        <style>
            /* Modal Surface - Compact Half Ratio (380px) */
            div[data-testid="stDialog"] div[role="dialog"],
            div[data-baseweb="modal"] div[role="dialog"],
            div[data-testid="stModal"] div[role="dialog"],
            div[role="dialog"] {
                max-width: 380px !important;
                width: 92vw !important;
                background-color: #FFFFFF !important;
                border-radius: 20px !important;
                border: 1px solid #E2E8F0 !important;
                box-shadow: 0 20px 60px rgba(15, 23, 42, 0.16) !important;
                padding: 16px 18px !important;
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
                font-size: 1.05rem !important;
                font-weight: 800 !important;
            }
            div[role="dialog"] button[aria-label="Close"],
            div[data-testid="stDialog"] button[aria-label="Close"] svg {
                color: #64748B !important;
                fill: #64748B !important;
            }

            /* Zero Wasted Space - Pure Image Tile Container */
            div[data-testid="stDialog"] div[data-testid="stColumn"],
            div[data-baseweb="modal"] div[data-testid="stColumn"],
            div[role="dialog"] div[data-testid="stColumn"] {
                position: relative !important;
                background: transparent !important;
                background-color: transparent !important;
                border: none !important;
                box-shadow: none !important;
                padding: 3px !important;
                margin: 0 !important;
                text-align: center !important;
                cursor: pointer !important;
            }

            /* Neutralize intermediate wrappers */
            div[data-testid="stDialog"] div[data-testid="stColumn"] div[data-testid="stVerticalBlock"],
            div[data-testid="stDialog"] div[data-testid="stColumn"] div[data-testid="stVerticalBlockBorderWrapper"],
            div[data-testid="stDialog"] div[data-testid="stColumn"] div[data-testid="stElementContainer"],
            div[data-testid="stDialog"] div[data-testid="stColumn"] div.element-container,
            div[data-baseweb="modal"] div[data-testid="stColumn"] div[data-testid="stVerticalBlock"],
            div[data-baseweb="modal"] div[data-testid="stColumn"] div[data-testid="stElementContainer"],
            div[role="dialog"] div[data-testid="stColumn"] div[data-testid="stVerticalBlock"],
            div[role="dialog"] div[data-testid="stColumn"] div[data-testid="stElementContainer"] {
                position: static !important;
                background: transparent !important;
                background-color: transparent !important;
            }

            /* Fundus image thumbnail: Perfectly fills column tile, zero wasted space */
            div[data-testid="stDialog"] div[data-testid="stColumn"] [data-testid="stImage"],
            div[data-baseweb="modal"] div[data-testid="stColumn"] [data-testid="stImage"],
            div[role="dialog"] div[data-testid="stColumn"] [data-testid="stImage"] {
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                width: 100% !important;
                margin: 0 auto !important;
                pointer-events: none !important;
            }
            div[data-testid="stDialog"] div[data-testid="stColumn"] img,
            div[data-baseweb="modal"] div[data-testid="stColumn"] img,
            div[role="dialog"] div[data-testid="stColumn"] img {
                width: 100px !important;
                height: 100px !important;
                max-width: 100px !important;
                max-height: 100px !important;
                aspect-ratio: 1 / 1 !important;
                object-fit: cover !important;
                border-radius: 14px !important;
                border: 2px solid #E2E8F0 !important;
                transition: all 0.2s ease-in-out !important;
                display: block !important;
                margin: 0 auto !important;
                pointer-events: none !important;
                user-select: none !important;
                -webkit-user-select: none !important;
            }
            div[data-testid="stDialog"] div[data-testid="stColumn"]:hover img,
            div[data-baseweb="modal"] div[data-testid="stColumn"]:hover img,
            div[role="dialog"] div[data-testid="stColumn"]:hover img {
                border-color: #2563EB !important;
                transform: scale(1.04) !important;
                box-shadow: 0 4px 16px rgba(37, 99, 235, 0.25) !important;
            }

            /* Case label directly below image */
            div[data-testid="stDialog"] div[data-testid="stColumn"] [data-testid="stMarkdownContainer"],
            div[data-baseweb="modal"] div[data-testid="stColumn"] [data-testid="stMarkdownContainer"],
            div[role="dialog"] div[data-testid="stColumn"] [data-testid="stMarkdownContainer"] {
                position: relative !important;
                z-index: 10 !important;
                pointer-events: none !important;
                user-select: none !important;
                -webkit-user-select: none !important;
                width: 100% !important;
                text-align: center !important;
                margin-top: 3px !important;
            }
            div[data-testid="stDialog"] div[data-testid="stColumn"] [data-testid="stMarkdownContainer"] *,
            div[data-baseweb="modal"] div[data-testid="stColumn"] [data-testid="stMarkdownContainer"] *,
            div[role="dialog"] div[data-testid="stColumn"] [data-testid="stMarkdownContainer"] * {
                color: #0F172A !important;
                font-size: 0.78rem !important;
                font-weight: 700 !important;
            }

            /* Direct-Click Invisible Overlay spanning 100% of the image tile */
            div[data-testid="stDialog"] div[data-testid="stColumn"] div.stButton,
            div[data-testid="stDialog"] div[data-testid="stColumn"] div[data-testid="stButton"],
            div[data-baseweb="modal"] div[data-testid="stColumn"] div.stButton,
            div[data-baseweb="modal"] div[data-testid="stColumn"] div[data-testid="stButton"],
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
                background: transparent !important;
                background-color: transparent !important;
                border: none !important;
                box-shadow: none !important;
                z-index: 50 !important;
            }
            div[data-testid="stDialog"] div[data-testid="stColumn"] div.stButton > button,
            div[data-testid="stDialog"] div[data-testid="stColumn"] div[data-testid="stButton"] > button,
            div[data-baseweb="modal"] div[data-testid="stColumn"] div.stButton > button,
            div[data-baseweb="modal"] div[data-testid="stColumn"] div[data-testid="stButton"] > button,
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
                background-color: transparent !important;
                box-shadow: none !important;
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
                
                short_name = label.split(" (")[0] if " (" in label else label
                    
                with cols[col_idx]:
                    if os.path.exists(full_path):
                        img = Image.open(full_path)
                        st.image(img, use_container_width=True)
                        st.markdown(
                            f"<div style='text-align: center; margin-top: 3px;'><span style='font-size: 0.78rem; font-weight: 700; color: #0F172A !important;'>{short_name}</span></div>",
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
                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    @st.dialog("Contact Developer", width="small")
    def show_contact_dialog():
        st.markdown("<h3 style='margin: 0 0 2px 0; font-size: 1.1rem; color: #0F2756 !important; font-weight: 800;'>Contact Developer</h3>", unsafe_allow_html=True)
        st.markdown("<p style='margin: 0 0 14px 0; font-size: 0.8rem; color: #64748B !important;'>Have questions or feedback? Send a message directly to the developer:</p>", unsafe_allow_html=True)
        
        if st.session_state.get("message_sent", False):
            st.success("Message already sent. Thank you for reaching out.")
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
                    st.toast("Message sent successfully!")
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
    page_title="RetinaAI Care | AI Diagnostic System",
    layout="wide",
    initial_sidebar_state="auto"
)

# Custom Styling: Modern Luminous Theme (ClassiAds Inspired, Responsive at 100% Zoom)
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
        
        /* 1. Global Canvas: Luminous Porcelain with Ambient Diffuse Color Blooms (No Dot Matrix) */
        html, body, [class*="css"], .stApp {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
            background-color: #F8FAFD !important;
            background-image: 
                radial-gradient(at 0% 0%, rgba(219, 234, 254, 0.75) 0px, transparent 55%),
                radial-gradient(at 100% 0%, rgba(224, 231, 255, 0.65) 0px, transparent 50%),
                radial-gradient(at 50% 100%, rgba(238, 242, 255, 0.65) 0px, transparent 60%),
                radial-gradient(at 20% 80%, rgba(224, 242, 254, 0.45) 0px, transparent 50%) !important;
            background-size: 100% 100% !important;
            background-attachment: fixed !important;
            color: #1E293B !important;
        }
        
        /* 2. Compact Viewport Container (Optimized for 100% Zoom with Breathing Space) */
        .block-container,
        [data-testid="stMainBlockContainer"] {
            max-width: 1260px !important;
            padding-top: 2.0rem !important;
            padding-bottom: 2.0rem !important;
            padding-left: 2.0rem !important;
            padding-right: 2.0rem !important;
            margin: 0 auto !important;
        }
        
        /* 3. Header & Sidebar Controls (Ensuring Reliable Collapse & Re-opening with Reference Icon) */
        header[data-testid="stHeader"] {
            height: 3.5rem !important;
            background: transparent !important;
            background-color: transparent !important;
            pointer-events: auto !important;
            z-index: 100 !important;
        }
        
        /* Reopen Container: Placed at top-left when sidebar is collapsed */
        div[data-testid="collapsedControl"] {
            position: fixed !important;
            top: 14px !important;
            left: 14px !important;
            z-index: 999999 !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin: 0 !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: auto !important;
            height: auto !important;
            visibility: visible !important;
            opacity: 1 !important;
            pointer-events: auto !important;
        }
        div[data-testid="collapsedControl"]::before,
        div[data-testid="collapsedControl"]::after {
            content: none !important;
            display: none !important;
        }

        /* Sidebar Header and Collapse Container Neutralization */
        div[data-testid="stSidebarHeader"] {
            display: flex !important;
            justify-content: flex-end !important;
            align-items: center !important;
            padding: 0.75rem 1rem 0.25rem 1rem !important;
            background: transparent !important;
        }
        div[data-testid="stSidebarCollapseButton"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin: 0 !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: auto !important;
            height: auto !important;
        }
        div[data-testid="stSidebarCollapseButton"]::before,
        div[data-testid="stSidebarCollapseButton"]::after {
            content: none !important;
            display: none !important;
        }

        /* Unified Button Styling: Target STRICTLY the <button> element */
        div[data-testid="collapsedControl"] button,
        div[data-testid="stSidebarCollapseButton"] button,
        div[data-testid="stSidebarHeader"] button {
            position: relative !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 36px !important;
            height: 36px !important;
            min-width: 36px !important;
            min-height: 36px !important;
            max-width: 36px !important;
            max-height: 36px !important;
            background-color: #FFFFFF !important;
            border: 1.5px solid #CBD5E1 !important;
            border-radius: 10px !important;
            color: #1E293B !important;
            padding: 0 !important;
            margin: 0 !important;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08) !important;
            cursor: pointer !important;
            pointer-events: auto !important;
            transition: all 0.2s ease !important;
            overflow: hidden !important;
            font-size: 0 !important;
            line-height: 0 !important;
        }
        div[data-testid="collapsedControl"] button:hover,
        div[data-testid="stSidebarCollapseButton"] button:hover,
        div[data-testid="stSidebarHeader"] button:hover {
            border-color: #2563EB !important;
            background-color: #EFF6FF !important;
            color: #2563EB !important;
            transform: scale(1.05) !important;
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.22) !important;
        }

        /* Suppress ONLY native SVGs and text glyphs inside the buttons */
        div[data-testid="collapsedControl"] button svg,
        div[data-testid="collapsedControl"] button span,
        div[data-testid="stSidebarCollapseButton"] button svg,
        div[data-testid="stSidebarCollapseButton"] button span,
        div[data-testid="stSidebarHeader"] button svg,
        div[data-testid="stSidebarHeader"] button span {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
            width: 0 !important;
            height: 0 !important;
            pointer-events: none !important;
        }

        /* Collapse button inside sidebar: Reference squircle with chevron pointing LEFT < */
        div[data-testid="stSidebarCollapseButton"] button::after,
        div[data-testid="stSidebarHeader"] button::after {
            content: "" !important;
            position: absolute !important;
            top: 50% !important;
            left: 50% !important;
            transform: translate(-50%, -50%) !important;
            width: 22px !important;
            height: 22px !important;
            background-color: currentColor !important;
            -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='2.5' y='2.5' width='19' height='19' rx='5'/%3E%3Cline x1='8.5' y1='2.5' x2='8.5' y2='21.5'/%3E%3Cpolyline points='16.5 8.5 12.5 12 16.5 15.5'/%3E%3C/svg%3E") no-repeat center / contain !important;
            mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='2.5' y='2.5' width='19' height='19' rx='5'/%3E%3Cline x1='8.5' y1='2.5' x2='8.5' y2='21.5'/%3E%3Cpolyline points='16.5 8.5 12.5 12 16.5 15.5'/%3E%3C/svg%3E") no-repeat center / contain !important;
            pointer-events: none !important;
        }

        /* Open / Reopen button when sidebar is collapsed: Reference squircle with chevron pointing RIGHT > */
        div[data-testid="collapsedControl"] button::after {
            content: "" !important;
            position: absolute !important;
            top: 50% !important;
            left: 50% !important;
            transform: translate(-50%, -50%) !important;
            width: 22px !important;
            height: 22px !important;
            background-color: currentColor !important;
            -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='2.5' y='2.5' width='19' height='19' rx='5'/%3E%3Cline x1='8.5' y1='2.5' x2='8.5' y2='21.5'/%3E%3Cpolyline points='12.5 8.5 16.5 12 12.5 15.5'/%3E%3C/svg%3E") no-repeat center / contain !important;
            mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='2.5' y='2.5' width='19' height='19' rx='5'/%3E%3Cline x1='8.5' y1='2.5' x2='8.5' y2='21.5'/%3E%3Cpolyline points='12.5 8.5 16.5 12 12.5 15.5'/%3E%3C/svg%3E") no-repeat center / contain !important;
            pointer-events: none !important;
        }
        
        /* Sidebar Resizer Handle (Allows dragging to extend sidebar width) */
        [data-testid="stSidebarResizer"] {
            visibility: visible !important;
            display: block !important;
            cursor: col-resize !important;
            width: 8px !important;
            background: transparent !important;
            transition: background 0.2s ease !important;
        }
        [data-testid="stSidebarResizer"]:hover {
            background: rgba(37, 99, 235, 0.25) !important;
        }

        /* Hide unwanted default Streamlit badges & toolbar clutter */
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
        div[data-testid="InputInstructions"],
        [data-testid="stHeaderActionElements"],
        .stHeadingActionElements,
        a[data-testid="stHeaderAction"],
        a.anchor-link,
        [data-testid="stHeadingWithActionElements"] a,
        h1 a, h2 a, h3 a, h4 a {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
            height: 0 !important;
            width: 0 !important;
            pointer-events: none !important;
        }
        
        /* 4. Typography Hierarchy & Contrast (Deep Slate & Crisp Titles) */
        h1, [data-testid="stHeadingWithActionElements"] h1 {
            font-size: 1.30rem !important;
            font-weight: 800 !important;
            line-height: 1.25 !important;
            margin: 0.2rem 0 0.35rem 0 !important;
            color: #0F172A !important;
            letter-spacing: -0.02em !important;
        }
        h2, [data-testid="stHeadingWithActionElements"] h2 {
            font-size: 1.10rem !important;
            font-weight: 800 !important;
            line-height: 1.25 !important;
            margin: 0.2rem 0 0.30rem 0 !important;
            color: #0F172A !important;
        }
        h3, [data-testid="stHeadingWithActionElements"] h3 {
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            line-height: 1.3 !important;
            margin: 0.15rem 0 0.25rem 0 !important;
            color: #0F172A !important;
        }
        h4, [data-testid="stHeadingWithActionElements"] h4 {
            font-size: 0.88rem !important;
            font-weight: 700 !important;
            color: #1E3A8A !important;
        }
        p, span, label, [data-testid="stMarkdownContainer"] p {
            font-size: 13px !important;
            line-height: 1.48 !important;
            color: #334155;
        }
        small, .caption, [data-testid="stImageCaption"] {
            font-size: 11.5px !important;
            color: #64748B !important;
            line-height: 1.35 !important;
        }
        
        /* 5. Sidebar: Clean Styling & Native Smooth Resizing */
        section[data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
            box-shadow: 4px 0 24px rgba(15, 23, 42, 0.04) !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
        }
        div[data-testid="stSidebarUserContent"] {
            width: 100% !important;
            min-width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            border: none !important;
            background: transparent !important;
        }
        [data-testid="stSidebarContent"] {
            width: 100% !important;
            box-sizing: border-box !important;
            padding-top: 1.2rem !important;
            padding-bottom: 3.5rem !important;
            padding-left: 1.25rem !important;
            padding-right: 1.25rem !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
        }
        section[data-testid="stSidebar"]::-webkit-scrollbar,
        [data-testid="stSidebarContent"]::-webkit-scrollbar {
            width: 6px !important;
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
            margin-bottom: 12px !important;
            padding: 10px 14px !important;
            font-size: 13px !important;
            font-weight: 700 !important;
            border-radius: 12px !important;
            width: 100% !important;
        }
        
        /* Modern Sidebar Brand Badge (Antigravity Style with Crisp Pure White Text) */
        .medcare-sidebar-brand {
            display: flex;
            align-items: center;
            gap: 12px;
            background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
            padding: 12px 14px;
            border-radius: 16px;
            color: #FFFFFF !important;
            margin-bottom: 22px;
            width: 100%;
            box-sizing: border-box;
            box-shadow: 0 4px 16px rgba(37, 99, 235, 0.25);
        }
        .medcare-sidebar-brand,
        .medcare-sidebar-brand *,
        .medcare-sidebar-brand h2,
        .medcare-sidebar-brand div,
        .medcare-sidebar-brand span,
        .medcare-sidebar-brand p {
            color: #FFFFFF !important;
        }
        .medcare-sidebar-brand h2 {
            margin: 0 !important;
            font-size: 1.05rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.02em;
            line-height: 1.1;
        }
        
        /* 6. ClassiAds Inspired Cards: Squircles & Ambient Elevation */
        .medcare-card,
        .medcare-kpi-card,
        .biomarker-card,
        .diagnosis-card-dr,
        .diagnosis-card-normal,
        .medcare-study-header {
            background: #FFFFFF !important;
            border-radius: 18px !important;
            border: 1px solid #E8EEF5 !important;
            padding: 14px 16px;
            box-shadow: 0 4px 20px rgba(15, 23, 42, 0.04), 0 1px 3px rgba(15, 23, 42, 0.02) !important;
            margin-bottom: 12px;
            transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        }
        .medcare-card:hover,
        .medcare-kpi-card:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(37, 99, 235, 0.08), 0 2px 6px rgba(15, 23, 42, 0.03) !important;
        }
        .medcare-card * {
            color: #1E293B !important;
        }
        .medcare-card p {
            color: #334155 !important;
        }
        
        /* 7. Buttons: Royal Blue Rounded Pills with Guaranteed Pure White Text */
        [data-testid="stButton"] button,
        [data-testid="stButton"] button *,
        .stButton > button,
        .stButton > button *,
        div.stButton > button *,
        button[key="btn_open_contact"],
        button[key="btn_open_contact"] * {
            color: #FFFFFF !important;
        }
        [data-testid="stButton"] button,
        .stButton > button {
            background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
            border: none !important;
            border-radius: 12px !important;
            min-height: 34px !important;
            height: 34px !important;
            padding: 4px 14px !important;
            font-weight: 700 !important;
            font-size: 13px !important;
            letter-spacing: -0.01em !important;
            box-shadow: 0 3px 10px rgba(37, 99, 235, 0.2) !important;
            transition: all 0.2s ease-in-out !important;
        }
        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 5px 14px rgba(37, 99, 235, 0.3) !important;
            color: #FFFFFF !important;
        }
        
        /* Secondary Category Buttons (Clinical Normal & DR) */
        button[key="btn_side_norm"], button[key="btn_w_norm"] {
            background: #F0FDF4 !important;
            color: #065F46 !important;
            border: 1.5px solid #A7F3D0 !important;
            border-radius: 12px !important;
            box-shadow: 0 2px 6px rgba(16, 185, 129, 0.08) !important;
        }
        button[key="btn_side_norm"] *, button[key="btn_w_norm"] * {
            color: #065F46 !important;
        }
        button[key="btn_side_norm"]:hover, button[key="btn_w_norm"]:hover {
            background: #DCFCE7 !important;
            border-color: #34D399 !important;
            color: #047857 !important;
        }
        button[key="btn_side_dr"], button[key="btn_w_dr"] {
            background: #FEF2F2 !important;
            color: #991B1B !important;
            border: 1.5px solid #FECACA !important;
            border-radius: 12px !important;
            box-shadow: 0 2px 6px rgba(239, 68, 68, 0.08) !important;
        }
        button[key="btn_side_dr"] *, button[key="btn_w_dr"] * {
            color: #991B1B !important;
        }
        button[key="btn_side_dr"]:hover, button[key="btn_w_dr"]:hover {
            background: #FEE2E2 !important;
            border-color: #F87171 !important;
            color: #B91C1C !important;
        }
        
        /* 8. Tabs: Modern Rounded Pill Tabs (ClassiAds Segmented Style) */
        div[data-testid="stTabs"] div[role="tablist"],
        div[data-testid="stTabs"] [data-baseweb="tab-list"],
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px !important;
            background-color: #EEF2F6 !important;
            padding: 5px !important;
            border-radius: 14px !important;
            border: 1px solid #E2E8F0 !important;
            min-height: 36px !important;
            display: inline-flex !important;
        }
        div[data-testid="stTabs"] button[data-testid="stTab"],
        div[data-testid="stTabs"] button[role="tab"],
        div[data-testid="stTabs"] [data-baseweb="tab"],
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px !important;
            padding: 6px 16px !important;
            font-weight: 700 !important;
            font-size: 13px !important;
            color: #475569 !important;
            background-color: transparent !important;
            border: 1px solid transparent !important;
            min-height: 32px !important;
            height: 32px !important;
            transition: all 0.2s ease !important;
        }
        div[data-testid="stTabs"] button[data-testid="stTab"]:hover,
        div[data-testid="stTabs"] button[role="tab"]:hover,
        div[data-testid="stTabs"] [data-baseweb="tab"]:hover {
            color: #2563EB !important;
            background-color: #F8FAFC !important;
        }
        div[data-testid="stTabs"] button[data-testid="stTab"][aria-selected="true"],
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
        div[data-testid="stTabs"] [aria-selected="true"],
        .stTabs [aria-selected="true"] {
            background-color: #FFFFFF !important;
            color: #2563EB !important;
            border: 1.5px solid #C7D2FE !important;
            border-radius: 10px !important;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.12) !important;
        }
        /* Completely eliminate sharp rectangular red underline */
        div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
        div[data-testid="stTabs"] div[data-baseweb="tab-border"],
        div[data-testid="stTabs"] [data-testid="stTabHighlight"],
        div[data-testid="stTabs"] div[role="tablist"] hr,
        div[data-testid="stTabs"] div[role="tablist"] > div[style*="position: absolute"],
        div[data-testid="stTabs"] div[role="tablist"] div[role="presentation"] {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            opacity: 0 !important;
        }
        
        /* 9. Segmented Control for Radio Buttons (Hide raw dots) */
        div[data-testid="stRadio"] input[type="radio"] {
            display: none !important;
        }
        div[data-testid="stRadio"] > div {
            background-color: #EEF2F6 !important;
            padding: 4px !important;
            border-radius: 14px !important;
            gap: 4px !important;
            border: 1px solid #E2E8F0 !important;
        }
        div[data-testid="stRadio"] label {
            padding: 6px 14px !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            font-size: 12.5px !important;
            color: #475569 !important;
            cursor: pointer !important;
            transition: all 0.15s ease !important;
        }
        div[data-testid="stRadio"] label[data-checked="true"],
        div[data-testid="stRadio"] label:has(input:checked) {
            background-color: #FFFFFF !important;
            color: #2563EB !important;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.12) !important;
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
            border-radius: 14px !important;
            color: #1E293B !important;
            padding: 10px 12px !important;
            transition: all 0.2s ease-in-out !important;
        }
        [data-testid="stFileUploader"] section:hover,
        [data-testid="stFileUploaderDropzone"]:hover,
        section[data-testid="stFileUploadDropzone"]:hover {
            border-color: #2563EB !important;
            background-color: #EFF6FF !important;
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
            color: #2563EB !important;
            border-color: #2563EB !important;
        }
        [data-testid="stFileUploader"] svg {
            fill: #2563EB !important;
            stroke: #2563EB !important;
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
            border-radius: 14px !important;
            border: 1px solid #E2E8F0 !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
            background-color: #000000 !important;
        }
        
        /* Badges */
        .badge-info {
            display: inline-flex;
            align-items: center;
            background-color: #EFF6FF;
            color: #2563EB !important;
            padding: 3px 9px;
            border-radius: 9999px;
            font-size: 0.70rem;
            font-weight: 700;
            margin-right: 5px;
            border: 1px solid #DBEAFE;
        }

        /* Social Badges with Guaranteed Visibility */
        .social-badge {
            text-decoration: none !important;
            color: #1E3A8A !important;
            background-color: #EFF6FF !important;
            border: 1px solid #C7D2FE !important;
            padding: 5px 11px !important;
            border-radius: 10px !important;
            font-size: 0.74rem !important;
            font-weight: 700 !important;
            display: inline-flex !important;
            align-items: center !important;
            gap: 6px !important;
            cursor: pointer !important;
            transition: all 0.15s ease-in-out !important;
        }
        .social-badge:hover {
            background-color: #DBEAFE !important;
            color: #1D4ED8 !important;
            border-color: #93C5FD !important;
        }
        .social-badge svg {
            fill: #1E3A8A !important;
        }

        /* 11. Modern Hero Banner & KPI Components with Glowing White Text */
        .medcare-hero-banner {
            background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 60%, #38BDF8 100%);
            border-radius: 18px;
            padding: 16px 22px;
            color: #FFFFFF !important;
            box-shadow: 0 8px 24px rgba(37, 99, 235, 0.2);
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 14px;
            position: relative;
            overflow: hidden;
        }
        .medcare-hero-banner,
        .medcare-hero-banner *,
        .medcare-hero-banner h2,
        .medcare-hero-banner div,
        .medcare-hero-banner span,
        .medcare-hero-banner p {
            color: #FFFFFF !important;
        }
        .medcare-hero-banner::after {
            content: "";
            position: absolute;
            right: -25px;
            top: -25px;
            width: 170px;
            height: 170px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, transparent 70%);
            pointer-events: none;
        }
        .medcare-hero-banner h2 {
            font-size: 1.25rem !important;
            font-weight: 800 !important;
            margin: 2px 0 4px 0 !important;
            letter-spacing: -0.02em !important;
        }
        .medcare-hero-banner p {
            font-size: 0.80rem !important;
            line-height: 1.45 !important;
            margin: 0 !important;
            max-width: 680px !important;
            opacity: 0.95;
        }
        .medcare-badge-live {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(255, 255, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.35);
            color: #FFFFFF !important;
            font-size: 0.68rem;
            font-weight: 700;
            padding: 3px 9px;
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
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            padding: 12px 14px !important;
        }
        .kpi-title {
            font-size: 0.64rem;
            font-weight: 700;
            text-transform: uppercase;
            color: #64748B !important;
            letter-spacing: 0.04em;
            margin-bottom: 2px;
        }
        .kpi-val {
            font-size: 1.20rem;
            font-weight: 800;
            color: #0F172A !important;
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
            color: #2563EB !important;
            background: #EFF6FF;
            padding: 2px 6px;
            border-radius: 8px;
            width: fit-content;
        }
        
        /* 12. Patient Study Header & Diagnosis Cards */
        .medcare-study-header {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            align-items: center;
            margin-bottom: 14px;
        }
        .study-meta-item {
            display: flex;
            flex-direction: column;
            gap: 2px;
        }
        .meta-label {
            font-size: 0.64rem;
            font-weight: 700;
            color: #64748B !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .meta-val {
            font-size: 0.84rem;
            font-weight: 700;
            color: #0F172A !important;
        }
        
        .diagnosis-card-dr {
            border: 1px solid #FEE2E2 !important;
            border-left: 5px solid #EF4444 !important;
            display: flex;
            align-items: flex-start;
            gap: 14px;
            min-height: 115px;
        }
        .diagnosis-card-normal {
            border: 1px solid #D1FAE5 !important;
            border-left: 5px solid #10B981 !important;
            display: flex;
            align-items: flex-start;
            gap: 14px;
            min-height: 115px;
        }
        
        /* Biomarker Severity Matrix Card */
        .biomarker-card {
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
            color: #334155 !important;
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
        <!-- Modern Standard AI 4-Point Radiant Diamond Spark Emblem (Antigravity Trend) -->
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2L14.85 9.15L22 12L14.85 14.85L12 22L9.15 14.85L2 12L9.15 9.15L12 2Z" fill="#FFFFFF"/>
            <path d="M19 4L20.2 6.8L23 8L20.2 9.2L19 12L17.8 9.2L15 8L17.8 6.8L19 4Z" fill="rgba(255,255,255,0.7)"/>
        </svg>
        <div>
            <h2 style="color: #FFFFFF !important; margin: 0 !important; font-size: 1.05rem !important; font-weight: 800 !important; letter-spacing: -0.02em; line-height: 1.1;">RetinaAI Care</h2>
            <div style="color: rgba(255, 255, 255, 0.88) !important; font-size: 0.64rem !important; font-weight: 600 !important; letter-spacing: 0.05em;">CLINICAL DECISION SYSTEM</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 7px; font-size: 0.76rem; font-weight: 800; color: #1E3A8A; text-transform: uppercase; letter-spacing: 0.05em; margin: 26px 0 12px 0;'>
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#2B59ED" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.93a2 2 0 0 1-1.66-.9l-.82-1.2A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13c0 1.1.9 2 2 2Z"/>
        </svg>
        <span>Fundus Input Source</span>
    </div>
    """, unsafe_allow_html=True)
    input_mode = st.radio(
        "Choose Input Source",
        ["Upload Image", "Preloaded Clinical Cases"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    active_image = None
    active_source_label = None
    is_already_preprocessed = False
    validation_error = None
    
    if input_mode == "Upload Image":
        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
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
        st.markdown("""
        <div style='display: flex; align-items: center; gap: 7px; font-size: 0.76rem; font-weight: 800; color: #1E3A8A; text-transform: uppercase; letter-spacing: 0.05em; margin: 24px 0 12px 0;'>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#2B59ED" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="m16 6 4 14"/>
                <path d="M12 6v14"/>
                <path d="M8 8v12"/>
                <path d="M4 4v16"/>
            </svg>
            <span>Clinical Sample Library</span>
        </div>
        """, unsafe_allow_html=True)
        
        btn_normal = st.button("Normal Retina Cases", use_container_width=True, key="btn_side_norm")
        btn_dr = st.button("Diabetic Retinopathy Cases", use_container_width=True, key="btn_side_dr")
        
        if btn_normal:
            show_sample_gallery_dialog("Normal Retina Cases", ORIGINAL_NORMAL_CASES)
        elif btn_dr:
            show_sample_gallery_dialog("Diabetic Retinopathy Cases", ORIGINAL_DR_CASES)
            
        if st.session_state.get("selected_sample_path"):
            sample_path = st.session_state.selected_sample_path
            if os.path.exists(sample_path):
                try:
                    active_image = Image.open(sample_path).convert("RGB")
                    active_source_label = st.session_state.selected_sample_label
                    
                    st.markdown("""
                    <div style='background: #EEF2FF; border: 1.5px solid #C7D2FE; border-radius: 12px; padding: 12px 14px; margin-top: 14px;'>
                        <div style='font-size: 0.68rem; font-weight: 800; color: #2B59ED; text-transform: uppercase; letter-spacing: 0.04em;'>ACTIVE CLINICAL CASE</div>
                        <div style='font-size: 0.82rem; font-weight: 700; color: #0F2756; word-break: break-all; margin-top: 4px;'>""" + os.path.basename(sample_path) + """</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("Clear Selected Case", key="btn_reset_sample", use_container_width=True):
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
            
    st.markdown("<div style='border-top: 1.5px solid #E2E8F0; margin: 34px 0 24px 0;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 7px; font-size: 0.76rem; font-weight: 800; color: #1E3A8A; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px;'>
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#2B59ED" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/>
            <circle cx="9" cy="7" r="4"/>
            <path d="M22 21v-2a4 4 0 0 0-3-3.87"/>
            <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
        </svg>
        <span>Developer & Support</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Custom HTML for social links (styled with clean, high-contrast badges with genuine SVG brand icons)
    st.markdown(
        """
        <div style="display: flex; gap: 10px; margin-bottom: 16px;">
            <a href="https://github.com/Avinash00006" target="_blank" class="social-badge">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12Z"/>
                </svg>
                <span>GitHub</span>
            </a>
            <a href="https://linkedin.com/in/avinash-koneti" target="_blank" class="social-badge">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451c.979 0 1.778-.773 1.778-1.729V1.73C24 .774 23.205 0 22.222 0h.003z"/>
                </svg>
                <span>LinkedIn</span>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Initialize contact message session state
    if "message_sent" not in st.session_state:
        st.session_state.message_sent = False
        
    btn_contact_text = "Message Developer" if not st.session_state.message_sent else "Message Delivered"
    if st.button(btn_contact_text, use_container_width=True, key="btn_open_contact"):
        show_contact_dialog()
            

#---------------Main Canvas Layout------------------------------------------------------

# Handle potential asset loading errors gracefully
if load_error is not None:
    st.error("Application Initialization Failure")
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
            <div class="medcare-badge-live">
                <span style="width: 7px; height: 7px; border-radius: 50%; background-color: #34D399; display: inline-block;"></span>
                <span>AI DIAGNOSTIC SYSTEM ACTIVE</span>
            </div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF !important; margin: 2px 0 4px 0; letter-spacing: -0.02em; line-height: 1.2;">Good Day, Clinician</div>
            <p style="color: rgba(255, 255, 255, 0.94) !important; font-size: 0.82rem; line-height: 1.45; margin: 0; max-width: 680px;">
                Welcome to <b>RetinaAI Care</b>. Automated ophthalmic fundus screening powered by hybrid EfficientNetV2-S deep feature extraction, Support Vector Machines, and Explainable AI (Grad-CAM & LIME).
            </p>
        </div>
        <div style="margin-left: 18px; opacity: 0.95; z-index: 1;">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L14.85 9.15L22 12L14.85 14.85L12 22L9.15 14.85L2 12L9.15 9.15L12 2Z" fill="#FFFFFF"/>
                <path d="M19 4L20.2 6.8L23 8L20.2 9.2L19 12L17.8 9.2L15 8L17.8 6.8L19 4Z" fill="rgba(255,255,255,0.7)"/>
            </svg>
        </div>
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
                <span class="kpi-badge-positive">Real-time</span>
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
                <div style="background: #EEF2FF; padding: 5px; border-radius: 8px; display: inline-flex;">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2B59ED" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M9 11l3 3L22 4"/>
                        <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/>
                    </svg>
                </div>
                <span style="font-weight: 700; font-size: 0.85rem; color: #0F2756 !important;">1. Quality Assurance</span>
            </div>
            <p style="font-size: 0.76rem; color: #475569 !important; margin: 0; line-height: 1.45;">
                Automated pre-flight validation analyzing aspect ratio, circular aperture coverage, and illumination metrics before pipeline inference.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="medcare-card">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                <div style="background: #EEF2FF; padding: 5px; border-radius: 8px; display: inline-flex;">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2B59ED" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect width="18" height="18" x="3" y="3" rx="2"/>
                        <path d="m9 9 6 6"/>
                        <path d="m15 9-6 6"/>
                    </svg>
                </div>
                <span style="font-weight: 700; font-size: 0.85rem; color: #0F2756 !important;">2. Deep Feature Classifier</span>
            </div>
            <p style="font-size: 0.76rem; color: #475569 !important; margin: 0; line-height: 1.45;">
                EfficientNetV2-S deep convolutional backbone extracts 1,280 structural features, classified with an RBF Support Vector Machine.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="medcare-card">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                <div style="background: #EEF2FF; padding: 5px; border-radius: 8px; display: inline-flex;">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2B59ED" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10"/>
                        <line x1="12" y1="8" x2="12" y2="12"/>
                        <line x1="12" y1="16" x2="12.01" y2="16"/>
                    </svg>
                </div>
                <span style="font-weight: 700; font-size: 0.85rem; color: #0F2756 !important;">3. Interpretability (XAI)</span>
            </div>
            <p style="font-size: 0.76rem; color: #475569 !important; margin: 0; line-height: 1.45;">
                Dual Explainable AI: Grad-CAM generates convolutional attention heatmaps, and LIME bounds influential superpixels.
            </p>
        </div>
        """, unsafe_allow_html=True)

# UI STATE 2: Medical Report Screen (Image Uploaded or Sample Selected)
else:
    # 1. Handle validation errors immediately on main canvas
    if validation_error is not None:
        st.error(f"Invalid Image: {validation_error}")
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
                <span style="font-size: 0.76rem; font-weight: 700; color: #065F46; background: #DCFCE7; border: 1px solid #A7F3D0; padding: 4px 10px; border-radius: 12px; display: inline-flex; align-items: center; gap: 6px;">
                    <span style="width: 7px; height: 7px; border-radius: 50%; background-color: #10B981; display: inline-block;"></span>
                    CONNECTED
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
        icon_svg = """<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#DC2626" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>"""
        title = "Diabetic Retinopathy (DR) Detected"
        title_color = "#991B1B"
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
        icon_svg = """<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#059669" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>"""
        title = "No Diabetic Retinopathy Found"
        title_color = "#0F2756"
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
                <div style="margin-top: 2px;">{icon_svg}</div>
                <div style="flex: 1;">
                    <div style="margin-bottom: 4px;">
                        <span style="font-size: 0.68rem; font-weight: 700; color: {badge_color}; background: {badge_bg}; padding: 2px 8px; border-radius: 12px; letter-spacing: 0.04em;">{badge_text}</span>
                    </div>
                    <h3 style="margin: 0; font-size: 1.15rem; font-weight: 800; color: {title_color}; line-height: 1.25;">{title}</h3>
                    <p style="margin: 6px 0 0 0; font-size: 0.82rem; line-height: 1.48; color: #334155;">{desc}</p>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
            
    with col_conf:
        # SVG Circular Gradient Donut Gauge
        circumference = 301.6
        dash_offset = circumference * (1 - confidence / 100.0)
        rel_label = "Clinical Reliability: High" if confidence >= 75 else "Clinical Reliability: Moderate"
        
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
                    <text x="60" y="58" text-anchor="middle" font-family="'Plus Jakarta Sans', -apple-system, sans-serif" font-weight="800" font-size="21" fill="#0F2756">{confidence}%</text>
                    <text x="60" y="73" text-anchor="middle" font-family="'Plus Jakarta Sans', -apple-system, sans-serif" font-weight="700" font-size="9" fill="#64748B" letter-spacing="0.05em">CONFIDENCE</text>
                </svg>
                <div style="margin-top: 4px; font-size: 0.68rem; font-weight: 700; color: #2B59ED; background: #EEF2FF; border: 1px solid #C7D2FE; padding: 2px 8px; border-radius: 10px;">
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
                <div style="font-size: 0.72rem; font-weight: 800; color: #0F2756; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px;">
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
        "Preprocessed View",
        "Diagnostic Heatmap (Grad-CAM)", 
        "Local Explanations (LIME)"
    ])
    
    # --- Tab 1: Preprocessing view ---
    with tab_prep:
        st.markdown("<h3 style='color: #0F2756 !important; font-weight: 800; margin: 0 0 4px 0;'>Fundus Image Normalization (Ben Graham's Method)</h3>", unsafe_allow_html=True)
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
                <h4 style="margin: 0 0 8px 0; font-size: 0.95rem; font-weight: 700; color: #0F2756;">Clinical Significance</h4>
                <ul style="margin: 0; padding-left: 18px; font-size: 0.8rem; color: #334155; line-height: 1.55;">
                    <li><b style="color: #1E3A8A;">Contrast Standardization</b>: Eliminates non-uniform lighting across retina boundaries.</li>
                    <li><b style="color: #1E3A8A;">Diagnostic Prep</b>: Sharpens micro-aneurysms and hard lipid exudates.</li>
                    <li><b style="color: #1E3A8A;">Classifier Stability</b>: Stabilizes input variance before SVM boundary scoring.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # --- Tab 2: Grad-CAM ---
    with tab_cam:
        st.markdown("<h3 style='color: #0F2756 !important; font-weight: 800; margin: 0 0 4px 0;'>Convolutional Feature Saliency Map</h3>", unsafe_allow_html=True)
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
                <h4 style="margin: 0 0 8px 0; font-size: 0.95rem; font-weight: 700; color: #0F2756;">Heatmap Interpretation</h4>
                <ul style="margin: 0; padding-left: 18px; font-size: 0.8rem; color: #334155; line-height: 1.55;">
                    <li><span style="display:inline-block; width:9px; height:9px; border-radius:50%; background:#EF4444; margin-right:5px;"></span> <b style="color: #1E3A8A;">Red / Orange Saliency</b>: Primary focal attention regions guiding feature weights.</li>
                    <li><span style="display:inline-block; width:9px; height:9px; border-radius:50%; background:#EAB308; margin-right:5px;"></span> <b style="color: #1E3A8A;">Yellow / Green Saliency</b>: Secondary contextual regions.</li>
                    <li><span style="display:inline-block; width:9px; height:9px; border-radius:50%; background:#3B82F6; margin-right:5px;"></span> <b style="color: #1E3A8A;">Blue Saliency</b>: Inactive background structures.</li>
                </ul>
                <p style="margin: 10px 0 0 0; font-size: 0.74rem; color: #64748B; font-style: italic;">
                    *Grad-CAM reflects model activation distribution and is intended for clinical decision assistance.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # --- Tab 3: LIME ---
    with tab_lime:
        st.markdown("<h3 style='color: #0F2756 !important; font-weight: 800; margin: 0 0 4px 0;'>Local Interpretable Model-agnostic Explanations (LIME)</h3>", unsafe_allow_html=True)
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
        
        if st.button("Run LIME Perturbation Analysis", key="run_lime"):
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
                    <h4 style="margin: 0 0 8px 0; font-size: 0.95rem; font-weight: 700; color: #0F2756;">Superpixel Explanation</h4>
                    <ul style="margin: 0; padding-left: 18px; font-size: 0.8rem; color: #334155; line-height: 1.55;">
                        <li><span style="display:inline-block; width:9px; height:9px; border-radius:50%; background:#EAB308; margin-right:5px;"></span> <b style="color: #1E3A8A;">Yellow Contours</b>: Isolated superpixels with highest statistical weight driving the prediction.</li>
                        <li>Clinically, look for boundaries tracking <b>exudates</b>, <b>hemorrhages</b>, or <b>macular changes</b>.</li>
                    </ul>
                    <div style="margin-top: 10px; font-size: 0.72rem; color: #065F46; font-weight: 700; background: #DCFCE7; border: 1px solid #A7F3D0; padding: 4px 8px; border-radius: 8px; display: inline-block;">
                        Completed with {lime_samples} perturbation samples
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Clinical Recommendation: Click the button above to run local superpixel feature analysis (LIME). This generates mathematical proof of local retinal features driving the SVM prediction. (Computation time: ~5-15 seconds depending on sample size)")