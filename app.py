import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# --- Page Config ---
st.set_page_config(
    page_title="Digit Recognizer AI",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%);
        color: #ffffff;
    }
    
    /* Title Styling */
    h1 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        color: #00d4ff;
        text-shadow: 0px 4px 10px rgba(0, 212, 255, 0.3);
        text-align: center;
        margin-bottom: 30px;
    }
    
    h2, h3 {
        color: #ffffff;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161625;
        border-right: 1px solid #333;
    }
    
    /* Card-like containers for columns */
    .css-1r6slb0, .stColumn {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #00d4ff 0%, #005bea 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 212, 255, 0.4);
    }
    
    /* Success Message */
    .stAlert {
        background-color: rgba(0, 255, 127, 0.1);
        border: 1px solid #00ff7f;
        color: #00ff7f;
        border-radius: 10px;
    }
    
    /* Metric Styling */
    div[data-testid="stMetricValue"] {
        font-size: 3rem;
        color: #00d4ff;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.h5')

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()

# --- Helpers ---
def preprocess_digit(img):
    """
    Robust preprocessing pipeline matching MNIST.
    """
    # 1. Grayscale & Invert (if needed)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Check if image is light background (like paper), then invert
    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)
    
    # 2. Thresholding (OTSU)
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Find Contours to center
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    digit = thresh[y:y+h, x:x+w]
    
    # 4. Resize maintaining aspect ratio
    if w > h:
        scale = 20.0 / w
        new_w = 20
        new_h = int(h * scale)
    else:
        scale = 20.0 / h
        new_h = 20
        new_w = int(w * scale)
        
    if new_w <= 0 or new_h <= 0: return None
        
    resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 5. Pad to 28x28 (Center of Mass is ideal, but center of box is good approx)
    padded = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # 6. Normalize
    processed_img = padded.astype('float32') / 255.0
    processed_img = processed_img.reshape(1, 28, 28, 1)
    
    return processed_img

# --- Main Layout ---
st.title("🔢 AI Digit Recognizer")
st.markdown("### Hand-drawn Digit Classification using CNN")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    input_method = st.radio("Choose Input:", ("✍️ Draw Digit", "📤 Upload Image"))
    st.markdown("---")
    st.info("💡 **Tip:** Draw clearly in the center for best results.")
    st.markdown("---")
    st.caption("v2.0 | Production Ready")

# Columns for Layout
col_input, col_process, col_result = st.columns([1.2, 0.8, 1.2])

final_image = None
raw_image = None

with col_input:
    st.markdown("#### 1. Input")
    if input_method == "✍️ Draw Digit":
        st.markdown("*Draw a digit (0-9) below:*")
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)", 
            stroke_width=15,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
            display_toolbar=True
        )
        if canvas_result.image_data is not None and np.sum(canvas_result.image_data[:, :, :3]) > 0:
            raw_image = canvas_result.image_data.astype('uint8')
            final_image = cv2.cvtColor(raw_image, cv2.COLOR_RGBA2GRAY)

    elif input_method == "📤 Upload Image":
        uploaded_file = st.file_uploader("Upload an image (PNG/JPG)", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image_pil = Image.open(uploaded_file).convert('L')
            st.image(image_pil, caption="Original Image", width=250)
            final_image = np.array(image_pil)

with col_process:
    st.markdown("#### 2. Process")
    if final_image is not None:
        processed_input = preprocess_digit(final_image)
        if processed_input is not None:
            # Display what the model sees
            st.image(
                processed_input.reshape(28, 28), 
                caption="Model Input (28x28)", 
                width=150,
                clamp=True,
                channels='GRAY'
            )
            st.success("Processed! ✅")
        else:
            st.warning("⚠️ Empty/Unclear")
    else:
        st.info("Waiting...")

with col_result:
    st.markdown("#### 3. Prediction")
    if final_image is not None and 'processed_input' in locals() and processed_input is not None:
        # Prediction
        with st.spinner("Analyzing..."):
            prediction = model.predict(processed_input)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction) * 100
        
        # Display Result Card
        st.markdown(f"""
        <div style="background-color: rgba(0, 212, 255, 0.1); padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #00d4ff;">
            <p style="color: #ccc; margin: 0;">Predicted Digit</p>
            <h1 style="font-size: 80px; margin: 0; color: #fff;">{predicted_class}</h1>
            <p style="color: #00d4ff; font-weight: bold;">{confidence:.2f}% Confidence</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("") # Spacer
        
        # Chart
        st.markdown("**Confidence Distribution:**")
        st.bar_chart(prediction[0], color="#00d4ff")
    else:
        st.write("Results will appear here.")

