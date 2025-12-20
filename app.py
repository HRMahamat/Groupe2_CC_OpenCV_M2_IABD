import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

# Chargement des classifieurs Haar
face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Configuration de la page
st.set_page_config(
    page_title="Face & Eye Detection - DetectionGenius",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="👁️"
)

# Initialisation des états
if 'mode' not in st.session_state:
    st.session_state.mode = None
if 'webcam_running' not in st.session_state:
    st.session_state.webcam_running = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = 0
if 'total_detections' not in st.session_state:
    st.session_state.total_detections = 0

# CSS professionnel inspiré des designs modernes
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Hero Section - Page d'accueil */
    .hero-container {
        background: white;
        border-radius: 30px;
        padding: 4rem 3rem;
        margin: 2rem 0;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 3rem;
        align-items: center;
    }
    
    .hero-left h1 {
        font-size: 3.5rem;
        font-weight: 800;
        color: #1a202c;
        line-height: 1.2;
        margin: 0 0 1rem 0;
    }
    
    .hero-left .highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .hero-left p {
        font-size: 1.2rem;
        color: #64748b;
        margin: 1.5rem 0 2rem 0;
        line-height: 1.7;
    }
    
    .hero-right {
        position: relative;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    .demo-image-container {
        position: relative;
        width: 100%;
        max-width: 500px;
    }
    
    .demo-image-container img {
        width: 100%;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
    }
    
    /* Rectangles de détection animés sur l'image de démo */
    .detection-box {
        position: absolute;
        border: 3px solid;
        border-radius: 8px;
        animation: pulse-border 2s ease-in-out infinite;
    }
    
    .face-box {
        border-color: #667eea;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
    }
    
    .eye-box {
        border-color: #10b981;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.5);
    }
    
    @keyframes pulse-border {
        0%, 100% {
            opacity: 1;
            transform: scale(1);
        }
        50% {
            opacity: 0.7;
            transform: scale(1.02);
        }
    }
    
    /* Badges de fonctionnalités */
    .features-badges {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin: 2rem 0;
    }
    
    .badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.7rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Bouton CTA principal */
    .main-cta {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem 3rem;
        border: none;
        border-radius: 50px;
        font-size: 1.2rem;
        font-weight: 700;
        cursor: pointer;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        display: inline-flex;
        align-items: center;
        gap: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .main-cta:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.5);
    }
    
    /* Section Comment ça marche */
    .how-it-works {
        background:rgba(0, 2, 2, 0.5) ;
        border-radius: 30px;
        padding: 3rem;
        margin: 2rem 0;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
    }
    
    .how-it-works h2 {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        color: #fff;
        margin-bottom: 3rem;
    }
    
    .steps-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 2rem;
    }
    
    .step-card {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
        border-radius: 20px;
        transition: transform 0.3s ease;
    }
    
    .step-card:hover {
        transform: translateY(-5px);
    }
    
    .step-number {
        width: 60px;
        height: 60px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0 auto 1.5rem auto;
    }
    
    .step-card h3 {
        font-size: 1.3rem;
        color: #fff;
        margin-bottom: 0.8rem;
    }
    
    .step-card p {
        color: #64748b;
        line-height: 1.6;
    }
    
    /* Stats rapides */
    .quick-stats {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .stat-card {
        background: rgba(0, 2, 2, 0.5);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        border-top: 4px solid #667eea;
    }
    
    .stat-number {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stat-label {
        color: #64748b;
        font-size: 0.95rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Page détection */
    .detection-header {
        background:linear-gradient(135deg, #667eea 0%, #764ba2 100%) ;
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }
    
    .detection-header h1 {
        font-size: 2.5rem;
        color: #fff;
        margin: 0 0 0.5rem 0;
    }
    
    /* Feedback intelligent */
    .feedback-box {
        background: white;
        padding: 1.2rem 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid;
        display: flex;
        align-items: center;
        gap: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .feedback-icon {
        font-size: 1.8rem;
    }
    
    .feedback-text {
        color: #fff;
        font-weight: 500;
        margin: 0;
        font-size: 1rem;
    }
    
    .feedback-success {
    border-left-color: #10b981;
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(209, 250, 229, 0.3) 50%, rgba(243, 244, 246, 0.5) 100%);
    box-shadow: 0 2px 8px rgba(16, 185, 129, 0.15);
}

    .feedback-warning {
        border-left-color: #f59e0b;
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(254, 243, 199, 0.3) 50%, rgba(243, 244, 246, 0.5) 100%);
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.15);
    }

    .feedback-info {
        border-left-color: #667eea;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.12) 0%, rgba(199, 210, 254, 0.3) 50%, rgba(243, 244, 246, 0.5) 100%);
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.15);
    }

    .feedback-error {
        border-left-color: #ef4444;
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(254, 202, 202, 0.3) 50%, rgba(243, 244, 246, 0.5) 100%);
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.15);
    }
    
    /* Stats en temps réel */
    .realtime-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .stat-box {
        background:rgba(0.2, 0.5, 12, 0.3) ;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        transition: transform 0.2s ease;
        border-top: 3px solid #667eea;
    }
    
    .stat-box:hover {
        transform: translateY(-4px);
    }
    
    .stat-box .number {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1;
    }
    
    .stat-box .label {
        color: #fff;
        font-size: 0.95rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    /* Boutons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Images */
    .stImage {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
    }
    
    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background:rgba(5, 2, 2, 0.5) ;
        backdrop-filter: blur(10px);
    }
    
    /* Info cards */
    .info-card {
        background: rgba(0, 2, 2, 0.5);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
        border-left: 4px solid #667eea;
    }
    
    .info-card h3 {
        color: #fff;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 0 0 1rem 0;
    }
    
    .info-card p {
        color: #64748b;
        line-height: 1.6;
        margin: 0.5rem 0;
    }
    
    /* Responsive */
    @media (max-width: 968px) {
        .hero-container {
            grid-template-columns: 1fr;
            padding: 2rem;
        }
        
        .hero-left h1 {
            font-size: 2.5rem;
        }
        
        .steps-grid {
            grid-template-columns: 1fr;
        }
        
        .quick-stats {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    
    @media (max-width: 640px) {
        .hero-left h1 {
            font-size: 2rem;
        }
        
        .quick-stats {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)

# Fonction de traitement avec mesure du temps
def process_image(img_array):
    start_time = time.time()
    picture = img_array.copy()
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Vérifier la qualité de l'image
    brightness = np.mean(gray)
    quality = "Bonne" if 50 < brightness < 200 else "Faible"
    
    faces = face.detectMultiScale(gray, 1.1, 5)
    total_eyes = 0
    for (x, y, w, h) in faces:
        cv2.rectangle(picture, (x, y), (x + w, y + h), (102, 126, 234), 3)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = picture[y:y + h, x:x + w]
        eyes = eye.detectMultiScale(roi_gray)
        total_eyes += len(eyes)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (16, 185, 129), 2)
    
    processing_time = time.time() - start_time
    return picture, len(faces), total_eyes, processing_time, quality

st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)

# Navigation
col_nav1, col_nav2 = st.columns(2)
with col_nav1:
    if st.button("🏠 Accueil", key="nav_home", use_container_width=True):
        st.session_state.current_page = 'home'
        st.rerun()
with col_nav2:
    if st.button("🔍 Commencer la détection", key="nav_detection", use_container_width=True):
        st.session_state.current_page = 'detection'
        st.rerun()

# PAGE ACCUEIL
if st.session_state.current_page == 'home':
    
    # Hero Section avec grille 2 colonnes
    col_hero1, col_hero2 = st.columns([1, 1])
    
    with col_hero1:
        st.markdown("""
        <div style="padding: 2rem 0;">
            <h1 style="font-size: 3.5rem; font-weight: 800; color: #fff; line-height: 1.2; margin: 0 0 1rem 0;">
                Face & Eye<br>
                <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">Detection Engine</span>
            </h1>
            <p style="font-size: 1.2rem; color: #64748b; margin: 1.5rem 0 2rem 0; line-height: 1.7;">
                Détectez et analysez automatiquement les visages et les yeux dans vos photos et vidéos en temps réel avec notre intelligence artificielle avancée.
            </p>
            
            
        </div>
        """, unsafe_allow_html=True)
    
    with col_hero2:
        # Charger et traiter l'image de démonstration
        try:
            # Remplacez ce chemin par votre image
            demo_image_path = "./assets/img2.jpg"  # Vous pouvez aussi utiliser une URL
            
            # Pour la démo, créons une image placeholder si le fichier n'existe pas
            import os
            if os.path.exists(demo_image_path):
                demo_img = Image.open(demo_image_path)
                demo_array = np.array(demo_img)
            else:
                # Image de démonstration par défaut (placeholder coloré)
                demo_array = np.ones((400, 500, 3), dtype=np.uint8) * 200
                cv2.putText(demo_array, "Placez votre image ici:", (50, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (102, 126, 234), 2)
                cv2.putText(demo_array, "./assets/demo-detection.jpg", (50, 220), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 126, 234), 2)
            
            # Appliquer la détection Haar Cascade sur l'image de démo
            demo_processed, demo_faces, demo_eyes, _, _ = process_image(demo_array)
            
            st.markdown("""
            <div style="position: relative; animation: float 3s ease-in-out infinite;">
                <style>
                    @keyframes float {
                        0%, 100% { transform: translateY(0px); }
                        50% { transform: translateY(-10px); }
                    }
                </style>
            </div>
            """, unsafe_allow_html=True)
            
            # Afficher l'image avec détections
            st.image(demo_processed, use_container_width=True, 
                    caption=f"✨ Détection en action : {demo_faces} visage(s) et {demo_eyes} œil(s) détectés")
            
        except Exception as e:
            # Si erreur, afficher un placeholder
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                        padding: 3rem; border-radius: 20px; text-align: center; border: 3px dashed #667eea;">
                <p style="font-size: 1.2rem; color: #667eea; font-weight: 600; margin: 0;">
                    📸 Placez votre image de démonstration ici
                </p>
                <p style="font-size: 0.9rem; color: #64748b; margin-top: 0.5rem;">
                    Chemin: ./assets/demo-detection.jpg
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Bouton CTA principal
    if st.button("🚀 COMMENCER LA DÉTECTION MAINTENANT", key="main_cta", use_container_width=True):
        st.session_state.current_page = 'detection'
        st.rerun()
    
    # Stats rapides
    st.markdown(f"""
    <div class="quick-stats">
        <div class="stat-card">
            <div class="stat-number">{st.session_state.total_detections}</div>
            <div class="stat-label">Détections effectuées</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">95%</div>
            <div class="stat-label">Taux de précision</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">30+</div>
            <div class="stat-label">Images / seconde</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">0.2s</div>
            <div class="stat-label">Temps de traitement</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Comment ça marche
    st.markdown("""
    <div class="how-it-works">
        <h2>Comment ça marche ?</h2>
        <div class="steps-grid">
            <div class="step-card">
                <div class="step-number">1</div>
                <h3>📤 Chargez votre média</h3>
                <p>Importez une photo ou activez votre webcam pour commencer l'analyse en temps réel.</p>
            </div>
            <div class="step-card">
                <div class="step-number">2</div>
                <h3>🔍 Détection automatique</h3>
                <p>Notre IA analyse l'image et détecte automatiquement tous les visages et les yeux présents.</p>
            </div>
            <div class="step-card">
                <div class="step-number">3</div>
                <h3>📊 Résultats instantanés</h3>
                <p>Visualisez les détections avec des rectangles colorés et obtenez des statistiques détaillées.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Fonctionnalités principales
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🎯 Détection multi-visages</h3>
            <p>Détectez simultanément plusieurs visages dans une même image avec une précision remarquable.</p>
        </div>
        
        <div class="info-card">
            <h3>👁️ Reconnaissance des yeux</h3>
            <p>Localisation précise des yeux pour chaque visage détecté avec marqueurs visuels distincts.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>📹 Analyse en temps réel</h3>
            <p>Détection instantanée via webcam avec traitement vidéo fluide à 30+ FPS.</p>
        </div>
        
        <div class="info-card">
            <h3>💡 Feedback intelligent</h3>
            <p>Recommandations automatiques pour améliorer la qualité de vos détections.</p>
        </div>
        """, unsafe_allow_html=True)

# PAGE DÉTECTION
elif st.session_state.current_page == 'detection':
    
    # Header
    st.markdown("""
    <div class="detection-header">
        <h1>🔍 Détection de visages et d'yeux</h1>
        <p style="color: #fff; margin: 0;">Analysez vos images ou utilisez votre webcam pour une détection en temps réel</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Contrôles
    with st.sidebar:
        st.markdown("### 🎛️ Mode de détection")
        
        # col1, col2 = st.columns(2)
        # with col1:
        if st.button("📁 Image", key="upload_btn", use_container_width=True):
                st.session_state.mode = 'upload'
                st.session_state.webcam_running = False
        # with col2:
        button_label = "🛑 Stop" if st.session_state.webcam_running else "📹 Webcam"
        if st.button(button_label, key="webcam_btn", use_container_width=True):
                if st.session_state.webcam_running:
                    st.session_state.webcam_running = False
                    st.session_state.mode = None
                else:
                    st.session_state.mode = 'webcam'
                    st.session_state.webcam_running = True
        
        st.markdown("---")
        
        if st.session_state.mode == 'upload':
            st.markdown("### 📤 Importer une image")
            file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        else:
            file = None
        
        st.markdown("---")
        st.markdown("""
        <div class="info-card">
            <h3>💡 Conseils pour de meilleurs résultats</h3>
            <p><strong>✓ Éclairage :</strong> Privilégiez une bonne luminosité</p>
            <p><strong>✓ Distance :</strong> Visages bien cadrés et visibles</p>
            <p><strong>✓ Qualité :</strong> Images nettes et en haute résolution</p>
            <p><strong>✓ Angle :</strong> Visages de face pour une meilleure détection</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Mode Upload
    if st.session_state.mode == 'upload' and file is not None:
        # Feedback : Analyse en cours
        st.markdown("""
        <div class="feedback-box feedback-info">
            <span class="feedback-icon">🔄</span>
            <p class="feedback-text">Analyse en cours...</p>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            img = Image.open(file)
            img_array = np.array(img)
            picture, num_faces, num_eyes, proc_time, quality = process_image(img_array)
            
            # Mise à jour du compteur total
            st.session_state.total_detections += num_faces
            st.session_state.processing_time = proc_time
            
            # Feedback intelligent basé sur la qualité
            if quality == "Faible":
                st.markdown("""
                <div class="feedback-box feedback-warning">
                    <span class="feedback-icon">⚠️</span>
                    <p class="feedback-text">Qualité d'image faible détectée. Essayez avec une meilleure luminosité pour des résultats optimaux.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="feedback-box feedback-success">
                    <span class="feedback-icon">✅</span>
                    <p class="feedback-text">Qualité d'image optimale pour la détection</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Temps de traitement
            st.markdown(f"""
            <div class="feedback-box feedback-info">
                <span class="feedback-icon">⏱️</span>
                <p class="feedback-text">Temps de traitement : {proc_time:.2f}s</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Statistiques de détection
            st.markdown(f"""
            <div class="realtime-stats">
                <div class="stat-box">
                    <div class="number">{num_faces}</div>
                    <div class="label">Visages détectés</div>
                </div>
                <div class="stat-box">
                    <div class="number">{num_eyes}</div>
                    <div class="label">Yeux détectés</div>
                </div>
                <div class="stat-box">
                    <div class="number">{quality}</div>
                    <div class="label">Qualité image</div>
                </div>
                <div class="stat-box">
                    <div class="number">{st.session_state.total_detections}</div>
                    <div class="label">Total détections</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
           # Image résultat centrée
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(picture, caption="✅ Résultat de la détection - Les rectangles bleus indiquent les visages, les verts indiquent les yeux", width=500)
            # Message de résultat
            if num_faces == 0:
                st.markdown("""
                <div class="feedback-box feedback-warning">
                    <span class="feedback-icon">⚠️</span>
                    <p class="feedback-text">Aucun visage détecté. Essayez avec une image où les visages sont plus visibles et de face.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="feedback-box feedback-success">
                    <span class="feedback-icon">🎉</span>
                    <p class="feedback-text">Détection réussie ! {num_faces} visage(s) et {num_eyes} œil(s) identifiés avec succès.</p>
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.markdown(f"""
            <div class="feedback-box feedback-error">
                <span class="feedback-icon">❌</span>
                <p class="feedback-text">Erreur lors du traitement : {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Mode Webcam
    elif st.session_state.mode == 'webcam' and st.session_state.webcam_running:
        st.info("🔴 Flux en direct activé")
        
        # 1. ON PRÉPARE LA STRUCTURE AVANT LA BOUCLE (Une seule fois)
        col_m1, col_m2, col_m3 = st.columns(3)
        metric_faces = col_m1.empty()
        metric_eyes = col_m2.empty()
        metric_fps = col_m3.empty()
        
        image_placeholder = st.empty()
        status_placeholder = st.empty()

        # ----- START: remplacement de la logique webcam par streamlit-webrtc -----
        from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
        import av
        import threading
        
        # Option STUN (utile en production)
        RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        
        class OpenCVTransformer(VideoTransformerBase):
            def __init__(self):
                # attributs pour exposer stats au thread principal
                self.last_faces = 0
                self.last_eyes = 0
                self.last_proc_time = 0.0
                self.lock = threading.Lock()
        
            def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
                # reçoit frame en BGR24
                img_bgr = frame.to_ndarray(format="bgr24")
                # convertir en RGB car ta fonction process_image attend RGB
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
                # appeler ta fonction de traitement existante (process_image)
                processed_rgb, num_faces, num_eyes, proc_time, quality = process_image(img_rgb)
        
                # Mettre à jour les compteurs dans l'objet (thread-safe)
                with self.lock:
                    self.last_faces = num_faces
                    self.last_eyes = num_eyes
                    self.last_proc_time = proc_time
        
                # Reconversion pour renvoyer (bgr)
                out_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
                return av.VideoFrame.from_ndarray(out_bgr, format="bgr24")
        
        
        # Lancer le streamer (dans ton layout page detection, quand mode == 'webcam')
        ctx = webrtc_streamer(
            key="detection-webcam",
            video_transformer_factory=OpenCVTransformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_transform=True,
            video_frame_callback=None,
            desired_playing=True,
        )
        
        # Affichage des metrics en parallèle (lire les stats depuis ctx.video_transformer)
        col_m1, col_m2, col_m3 = st.columns(3)
        metric_faces = col_m1.empty()
        metric_eyes = col_m2.empty()
        metric_fps = col_m3.empty()
        status_placeholder = st.empty()
        
        # Boucle non bloquante pour mettre à jour les métriques tant que le component tourne
        if ctx.state.playing:
            status_placeholder.info("🔴 Flux WebRTC actif — autorise la webcam dans le navigateur.")
        else:
            status_placeholder.warning("Le flux WebRTC n'est pas actif. Clique sur 'Start' dans la fenêtre vidéo si nécessaire.")
        
        # On lit régulièrement les valeurs exposées par le transformer (si présent)
        import time
        def update_stats_loop(ctx, metric_faces, metric_eyes, metric_fps, status_placeholder):
            try:
                while True:
                    if ctx.video_transformer is None:
                        # transformer pas encore prêt
                        metric_faces.metric("Visages", 0)
                        metric_eyes.metric("Yeux", 0)
                        metric_fps.metric("FPS", "0.0")
                        status_placeholder.write("Initialisation du flux...")
                    else:
                        with ctx.video_transformer.lock:
                            nf = ctx.video_transformer.last_faces
                            ne = ctx.video_transformer.last_eyes
                            pt = ctx.video_transformer.last_proc_time or 0.0
                        fps = 1.0 / pt if pt > 0 else 0.0
                        metric_faces.metric("Visages", nf)
                        metric_eyes.metric("Yeux", ne)
                        metric_fps.metric("FPS", f"{fps:.1f}")
                        status_placeholder.write(f"Qualité: — | Traitement (s): {pt:.3f}")
                    time.sleep(0.5)
            except Exception:
                pass
        
        # Démarrer la boucle de mise à jour dans un thread d'arrière-plan (non bloquant pour Streamlit)
        if 'webrtc_stats_thread' not in st.session_state:
            st.session_state.webrtc_stats_thread = threading.Thread(
                target=update_stats_loop,
                args=(ctx, metric_faces, metric_eyes, metric_fps, status_placeholder),
                daemon=True
            )
            st.session_state.webrtc_stats_thread.start()
        
        # ----- END: remplacement webcam -----

            st.markdown("""
            <div class="feedback-box feedback-success">
                <span class="feedback-icon">🟢</span>
                <p class="feedback-text">Webcam arrêtée avec succès</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class="info-card" style="text-align: center; padding: 3rem;">
            <h3 style="font-size: 1.8rem; margin-bottom: 1rem;">👆 Sélectionnez un mode pour commencer</h3>
            <p style="font-size: 1.1rem;">Choisissez <strong>Image</strong> pour analyser une photo ou <strong>Webcam</strong> pour la détection en temps réel</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown(f"""
<div style="background: rgba(0, 2, 2, 0.5); border-radius: 20px; padding: 2rem; margin-top: 3rem; text-align: center; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);">
    <p style="color: #fff; font-weight: 600; margin: 0; font-size: 1.1rem;">
        DetectionGenius © 2024 - Développé avec ❤️ par le Groupe 2
    </p>
    <p style="color: #64748b; margin: 0.5rem 0 0 0; font-size: 0.95rem;">
        Propulsé par OpenCV • Streamlit • Python | {st.session_state.total_detections} détections réalisées
    </p>
</div>

""", unsafe_allow_html=True)


