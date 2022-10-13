import streamlit as st
import requests
import numpy as np
import cv2
from PIL import Image
import glob


st.title("Future Vision Transport 🚘")
st.header("Prédiction d'une image via appel API")
st.info("""
La segmentation d'une image est prédite grâce à un model de deep learning. Le
modèle est utilisé par l'API.
On peut tester le modèle grâce à une liste d'image de test que nous disposons.
Il suffit de choisir l'id de l'image grâce au menu glissant ci-dessous.

Une fois l'id choisi, cliquer sur le bouton "Prédire" pour afficher le masque
prédit.
""")


val_imgs = glob.glob("images/color/*.png")
val_masks = glob.glob("images/mask/*.png")

image_id = st.slider("Choisissez l'id de l'image", 0, 499, 1)

color_image = Image.open(val_imgs[image_id])
mask_image = Image.open(val_masks[image_id])

col1, col2 = st.columns(2)
with col1:
    st.subheader("Image d'entrée en couleur")
    st.image(color_image)

with col2:
    st.subheader("Masque de l'image")
    st.image(mask_image)
    

url = "http://127.0.0.1:5000/predict"
content_type = 'image/png'
headers = {'content-type': content_type}


def rgb_seg_img(seg_arr, n_classes):
    
    class_colors = {
        0:(0,0,0),        # void
        1:(128, 64, 128), # flat
        2:(102,102,156),  # construction
        3:(153,153,153),  # object
        4:(107, 142, 35), # nature
        5:(70,130,180),   # sky
        6:(255, 0, 0),    # human
        7:(0, 0, 142)     # vehicle
    }
    
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]

    seg_img = np.zeros((output_height, output_width, 3))

    for c in range(n_classes):
        seg_arr_c = seg_arr[:, :] == c
        seg_img[:, :, 0] += ((seg_arr_c) * (class_colors[c][0])).astype('uint8') # R
        seg_img[:, :, 1] += ((seg_arr_c) * (class_colors[c][1])).astype('uint8') # G
        seg_img[:, :, 2] += ((seg_arr_c) * (class_colors[c][2])).astype('uint8') # B

    return seg_img.astype('uint8')


def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


local_css("style.css")
legend = """<div>
<span class='highlight bold void'>        </span>
<span>void</span>
</div>

<div>
<span class='highlight bold flat'>        </span>
<span>flat</span>
</div>

<div>
<span class='highlight bold construction'>        </span>
<span>construction</span>
</div>

<div>
<span class='highlight bold object'>        </span>
<span>object</span>
</div>

<div>
<span class='highlight bold nature'>        </span>
<span>nature</span>
</div>

<div>
<span class='highlight bold sky'>        </span>
<span>sky</span>
</div>

<div>
<span class='highlight bold human'>        </span>
<span>human</span>
</div>

<div>
<span class='highlight bold vehicle'>        </span>
<span>vehicle</span>
</div>
"""

if st.button("Prédire"):
    with st.spinner("Prédiction en cours..."):
        img = cv2.imread(val_imgs[image_id])
        # encode image as jpeg
        _, img_encoded = cv2.imencode('.png', img)

        # send http request with image and receive response
        response = requests.post(url, data=img_encoded.tobytes(), headers=headers)

        # decode response
        pred_data = response.json()["prediction"]
        pred_data = np.array(pred_data)
        pred_data = rgb_seg_img(pred_data, 8)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Masque prédit")
            st.text(" ")
            st.image(pred_data)
        with col4:
            st.subheader("Légende")
            st.markdown(legend, unsafe_allow_html=True)

# https://github.com/Abdess/Future-Vision-Transport-API/blob/e59b329b214ec18151d4b50168dcbf3bf6921c48/app/scripts/utils.py