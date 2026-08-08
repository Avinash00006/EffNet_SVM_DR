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

#---------------Streamlit UI Page Settings---------------------------------------------
st.set_page_config(
    page_title="Retinal diagnostics AI Workstation",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Plus Jakarta Sans, medical dark-mode console adaptions, and custom cards)
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
        
        /* Font overrides */
        html, body, [class*="css"], .stApp {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }
        
        /* Reduce empty padding at the top of the main area */
        .block-container {
            padding-top: 1.8rem !important;
            padding-bottom: 1.5rem !important;
        }
        
        /* Transparent header containing toggle button */
        header[data-testid="stHeader"] {
            background: transparent !important;
            background-color: transparent !important;
        }
        
        /* Headers styling */
        h1, h2, h3, h4 {
            font-weight: 700 !important;
        }
        
        /* Custom sidebar panel styling */
        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.2);
        }
        
        /* Badges styling */
        .badge-info {
            display: inline-block;
            background-color: rgba(128, 128, 128, 0.15);
            color: inherit;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 5px;
            border: 1px solid rgba(128, 128, 128, 0.2);
        }

        /* Hide Streamlit default Deploy, main menu, and decoration bar */
        .stDeployButton {
            display: none !important;
        }
        #MainMenu {
            visibility: hidden !important;
        }
        footer {
            visibility: hidden !important;
        }
        div[data-testid="stDecoration"] {
            display: none !important;
        }
        
        /* Hide the "Press Enter to apply" instruction overlays from inputs */
        div[data-testid="InputInstructions"] {
            display: none !important;
        }

        
        /* Custom image borders for clinical workstation look */
        .img-container img {
            border: 2px solid rgba(128, 128, 128, 0.3);
            border-radius: 8px;
            background-color: black;
        }
    </style>
""", unsafe_allow_html=True)

#---------------Load Models & Check Errors----------------------------------------------
full_model, feature_extractor, scaler, svm, load_error = load_models()

#---------------Sidebar Layout (Ophthalmic Control Console)----------------------------
with st.sidebar:
    st.markdown("## 👁️ Ophthalmic Console")
    st.markdown("<span style='font-size: 0.8rem; color:#64748B; font-weight:600;'>PACS DECISION SUPPORT SYSTEM</span>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.subheader("📁 Fundus Photography Upload")
    uploaded_file = st.file_uploader(
        "Upload a retinal fundus image (.jpg, .jpeg, .png)",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed"
    )
    
    # State placeholders
    is_already_preprocessed = False
    validation_error = None
    
    # Run validation immediately in sidebar background to update controls
    if uploaded_file is not None:
        try:
            original_image = Image.open(uploaded_file).convert("RGB")
            is_valid, validation_msg, is_already_preprocessed = validate_fundus_image(original_image)
            if not is_valid:
                validation_error = validation_msg
        except Exception as e:
            validation_error = f"Failed to parse image file: {str(e)}"
            
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

# UI STATE 1: Welcome Screen (Awaiting File Upload)
if uploaded_file is None:
    st.title("👁️ Retinal AI Diagnostics Workstation")
    st.markdown(
        '<span class="badge-info">Clinical Decision Support Tool</span>'
        '<span class="badge-info">EfficientNetV2-S + SVM Hybrid</span>'
        '<span class="badge-info">XAI Layers Enabled</span>', 
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Styled welcome card
    st.markdown(
        """
        <div style="border: 1px solid rgba(128,128,128,0.2); border-radius: 12px; padding: 24px; background-color: rgba(128,128,128,0.03); margin-bottom: 25px;">
            <h3 style="margin-top:0; font-size:1.4rem;">💻 Medical Imaging Console Status: Ready</h3>
            <p style="margin: 0; font-size: 1rem; color: inherit; line-height: 1.6;">
                This workstation provides automated screening and explanation interfaces for fundus photography.
                The system analyzes structural retinal details (vessels, macula, and optic disc) to evaluate risk indicators for 
                <b>Diabetic Retinopathy (DR)</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # 3-column workflow overview
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.markdown("### 📋 1. Quality Control")
            st.markdown(
                "Ensures uploaded photographs comply with clinical standards by analyzing aspect ratio, circular aperture bounds, and color distributions."
            )
    with col2:
        with st.container(border=True):
            st.markdown("### 🧠 2. Classification")
            st.markdown(
                "Uses a PyTorch convolutional network backbone to extract deep features, passing them to a Support Vector Machine (SVM) to locate decision boundaries."
            )
    with col3:
        with st.container(border=True):
            st.markdown("### 🔬 3. Interpretation")
            st.markdown(
                "Supports visual confirmation with saliency maps (Grad-CAM) to highlight model focus, and superpixel analysis (LIME) to define features."
            )
            
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.success("👈 Upload a retinal fundus image in the sidebar panel to initialize diagnostics.")

# UI STATE 2: Medical Report Screen (Image Uploaded & Validated)
else:
    # 1. Handle validation errors immediately on main canvas
    if validation_error is not None:
        st.error(f"❌ **Invalid Image**: {validation_error}")
        st.stop()
        
    # 2. Render Medical Report Header (Hiding welcome information and headers)
    st.markdown("### 👁️ Retinal AI Workstation // Diagnostic Report")
    
    # Structured Patient / Study Metadata Card
    st.markdown(
        """
        <div style="display: flex; flex-wrap: wrap; gap: 24px; padding: 14px; border: 1px solid rgba(128,128,128,0.2); border-radius: 8px; background-color: rgba(128,128,128,0.05); margin-bottom: 20px; font-size: 0.8rem; letter-spacing: 0.02em;">
            <div><span style="color: #64748B; font-weight: 700;">STUDY TYPE:</span> <span style="font-weight: 500;">Ophthalmic Fundus Photography</span></div>
            <div><span style="color: #64748B; font-weight: 700;">MODEL SPECIFICATION:</span> <span style="font-weight: 500;">EfficientNetV2-S + SVM</span></div>
            <div><span style="color: #64748B; font-weight: 700;">CRITERIA CALIBRATION:</span> <span style="font-weight: 500;">APTOS 2019 Calibration Scale</span></div>
            <div><span style="color: #64748B; font-weight: 700;">PAC WORKSTATION:</span> <span style="color: #10B981; font-weight: 700;">🟢 CONNECTED</span></div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Diagnostic Processing Pipeline
    original_image = Image.open(uploaded_file).convert("RGB")
    
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
    
    # Section 1: Clinical Diagnosis Card & Custom Gauge (Premium Workstation look)
    col_status, col_conf = st.columns([3, 1])
    
    with col_status:
        # Custom HTML diagnostics panel
        if prediction == 1:
            icon = "🚨"
            border_color = "#EF4444"
            bg_color = "rgba(239, 68, 68, 0.12)"
            text_color = "#EF4444"
            title = "Diabetic Retinopathy (DR) Detected"
            desc = "The hybrid inference pipeline identified microvascular abnormalities, microaneurysms, or hemorrhagic patterns in the fundus photography. Clinical evaluation and optical coherence tomography (OCT) follow-up are recommended."
        else:
            icon = "✅"
            border_color = "#10B981"
            bg_color = "rgba(16, 185, 129, 0.12)"
            text_color = "#10B981"
            title = "No Diabetic Retinopathy Found"
            desc = "Retinal structure mapping presents normal vascular distribution. Exudates, hemorrhages, and microaneurysms are absent or lie within SVM decision margins. Periodic screening is advised."
            
        st.markdown(
            f"""
            <div style="border-left: 6px solid {border_color}; background-color: {bg_color}; padding: 22px; border-radius: 8px; display: flex; align-items: flex-start; gap: 16px; min-height: 120px;">
                <span style="font-size: 2.2rem; line-height: 1.1;">{icon}</span>
                <div>
                    <h3 style="margin: 0; color: {text_color}; font-size: 1.25rem; font-weight: 700; line-height: 1.2;">{title}</h3>
                    <p style="margin: 8px 0 0 0; font-size: 0.95rem; line-height: 1.5; color: inherit;">{desc}</p>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
            
    with col_conf:
        # SVG Circular Gauge for clinical feel
        stroke_color = "#EF4444" if prediction == 1 else "#10B981"
        circumference = 314.16
        dash_offset = circumference * (1 - confidence / 100.0)
        
        st.markdown(
            f"""
            <div style="border: 1px solid rgba(128,128,128,0.2); border-radius: 8px; padding: 12px; background-color: rgba(128,128,128,0.03); display: flex; flex-direction: column; align-items: center; justify-content: center; height: 120px;">
                <svg width="78" height="78" viewBox="0 0 120 120" style="margin-top: -5px;">
                    <circle cx="60" cy="60" r="50" fill="none" stroke="rgba(128, 128, 128, 0.15)" stroke-width="10"></circle>
                    <circle cx="60" cy="60" r="50" fill="none" stroke="{stroke_color}" stroke-width="10" 
                            stroke-dasharray="{circumference}" stroke-dashoffset="{dash_offset}" stroke-linecap="round"
                            transform="rotate(-90 60 60)"></circle>
                    <text x="60" y="66" text-anchor="middle" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-weight="700" font-size="20" fill="currentColor">{confidence}%</text>
                </svg>
                <div style="font-size: 0.7rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em; color: #64748B; margin-top: 5px; line-height: 1;">Confidence</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 2: Clinical Dashboard Tabs (Visualizations & Explainability)
    tab_prep, tab_cam, tab_lime = st.tabs([
        "⚙️ Preprocessed View",
        "📊 Diagnostic Heatmap (Grad-CAM)", 
        "🔬 Local Explanations (LIME)"
    ])
    
    # --- Tab 1: Preprocessing view ---
    with tab_prep:
        st.markdown("### ⚙️ Fundus Image Normalization (Ben Graham's Method)")
        st.write(
            "Retinal fundus photographs often exhibit light reflections and poor contrast. "
            "Applying local Gaussian mean subtraction corrects spatial color and emphasizes vessels/lesions."
        )
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_p1, col_p2, col_p_desc = st.columns([1, 1, 1.2])
        with col_p1:
            st.image(original_image, caption="Original Photography", width=360)
        with col_p2:
            st.image(preprocessed_img, caption="Graham Contrast-Enhanced", width=360)
        with col_p_desc:
            st.markdown("#### Clinical Significance")
            st.info("""
            * **Contrast Standardization**: Illuminates capillary structures that are hidden by poor focus or shadows.
            * **Diagnostic Prep**: Highlights micro-aneurysms (appearing as sharp dark dots) and hard lipid exudates.
            * **Stability**: Neutralizes lighting variations, ensuring consistent SVM classifier inputs.
            """)

    # --- Tab 2: Grad-CAM ---
    with tab_cam:
        st.markdown("### 🔎 Convolutional Feature Saliency Map")
        st.write(
            "Grad-CAM calculates gradients relative to the final convolutional block of the EfficientNetV2-S "
            "model to visualize where the deep neural network focused while extracting features."
        )
        st.markdown("<br>", unsafe_allow_html=True)
        
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
            st.image(original_image, caption="Original Photography", width=360)
        with col_img2:
            st.image(overlay, caption="Grad-CAM Saliency Overlay", width=360)
        with col_desc:
            st.markdown("#### Heatmap Interpretation")
            st.info("""
            * 🔴 **Red / Orange Saliency**: High-importance nodes (vessels or lesions that dominated feature extraction).
            * 🟡 **Yellow / Green Saliency**: Moderate-importance regions.
            * 🔵 **Blue Saliency**: Background layers that had zero influence on the model.
            
            *The Grad-CAM map indicates model attention only, and should not be confused with exact lesion boundaries.*
            """)

    # --- Tab 3: LIME ---
    with tab_lime:
        st.markdown("### 🧠 Local Interpretable Model-agnostic Explanations (LIME)")
        st.write(
            "LIME segments the fundus image into superpixels (homogeneous regions), perturbs them "
            "randomly, and queries the SVM to see how predictions shift. Highlighted sections indicate the features "
            "that contributed most positive weight to the final decision."
        )
        st.markdown("<br>", unsafe_allow_html=True)
        
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
                st.image(lime_result, caption="LIME Highlighted Superpixels", width=360)
            with col_lime_info:
                st.markdown("#### Clinical Interpretation")
                st.success("""
                * **Highlighted Boundaries (Yellow)**: The specific superpixel regions containing the highest statistical weight driving the prediction.
                * Clinically, look for LIME boundaries tracking:
                  * **Exudates**: Bright yellowish clumps of fats/proteins.
                  * **Hemorrhages**: Dark bleeding spots (dot-blot).
                  * **Macula/Optic Disc boundaries** where structural changes have occurred.
                """)
                st.info(f"Perturbation sampling successfully completed with {lime_samples} runs.")
        else:
            st.info("💡 **Clinical Recommendation**: Click the button above to run local superpixel feature analysis (LIME). This generates mathematical proof of local retinal features driving the SVM prediction. (Computation time: ~5-15 seconds depending on sample size)")