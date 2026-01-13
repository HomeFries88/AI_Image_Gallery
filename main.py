import cv2
import numpy as np
import streamlit as st
import pandas as pd
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image
import io
import time

def load_model():
    model = MobileNetV2(weights = "imagenet")
    return model


def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis = 0)
    return img


def classify_image(model, image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top = 5)[0]
        return decoded_predictions
    except Exception as e:
        st.error(f"ERROR CLASSIFYING IMAGE: {str(e)}")
        return None
    

def main():
    st.set_page_config(page_title = "AI Image Gallery", page_icon = "📷", layout = "wide")
    st.title("AI Image Gallery Manager")
    st.write("Upload Multiple Images and the AI Will Automatically Sort Them!")
    
    @st.cache_resource
    def load_cached_model():
        return load_model()
    model = load_cached_model()

    if "images" not in st.session_state:
        st.session_state.images = []
    if "next_id" not in st.session_state:
        st.session_state.next_id = 0
    if "upload_key" not in st.session_state:
        st.session_state.upload_key = 0

    pages = [
        "Upload & Classify",
        "Gallery",
        "Search & Export"
    ]

    page = st.sidebar.selectbox("Navigate", pages)
    st.sidebar.markdown("---")

    if st.sidebar.button("Clear Gallery"):
        st.session_state.images = []
        st.session_state.next_id = 0
        st.sidebar.success("Gallery Cleared!")

    if page == "Upload & Classify":
        st.subheader("Add Images To Your Gallery!")

        uploaded_files = st.file_uploader(
            "Upload Images",
            type = ["jpg", "jpeg", "png"],
            accept_multiple_files = True,
            key = f"uploader_{st.session_state.upload_key}"
        )

        if uploaded_files:
            total_files = len(uploaded_files)
            progress_bar = st.progress(0)
            status_text = st.empty()
            added_count = 0

            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name} ({i+1}/{total_files})...")
                image = Image.open(file)
                predictions = classify_image(model, image)

                if predictions:
                    top_labels = [label for _, label, _ in predictions]
                    st.session_state.images.append({
                        "id": st.session_state.next_id,
                        "name": file.name,
                        "image": image,
                        "labels": top_labels,
                        "predictions": predictions,
                        "timestamp": time.strftime("%Y-%M-%d %H:%M:%S"),
                    })
                    
                    st.session_state.next_id += 1
                    added_count += 1

                progress_bar.progress((i + 1)/total_files)
            status_text.text("")
            if added_count > 0:
                st.success(f"{added_count} images uploaded & classified!")
                st.session_state.upload_key += 1
                st.rerun()


        if st.session_state.images:
            st.subheader("Recently Added")
            cols = st.columns(3)

            for idx, item in enumerate(st.session_state.images[-3:]):
                with cols[idx % 3]:
                    st.image(item["image"], use_column_width = True)
                    st.caption(item["name"])

                    for _, label, score in item["predictions"][:3]:
                        st.write(f"{label}: {score:.2%}")
    
    elif page == "Gallery":
        st.subheader("Your AI Tagged Image Gallery")

        if not st.session_state.images:
            st.info("No Images Uploaded So Far 🫠")
        else:
            cols = st.columns(3)

            for idx, item in enumerate(st.session_state.images):
                with cols[idx % 3]:
                    st.image(item["image"], use_container_width=True) # Note: use_column_width is deprecated in newer Streamlit
                    st.caption(f"{item['name']} (ID: {item['id']})") # Fixed quotes
                    st.write(f"Added: {item['timestamp']}")
                    
                    with st.expander("View Tags"):
                        for _, label, score in item["predictions"]:
                            st.write(f"**{label}**: {score:.2%}")
                    
                    if st.button("Remove", key = f"remove_{item['id']}"):
                        st.session_state.images = [i for i in st.session_state.images if i["id"] != item["id"]]
                        st.rerun()

    elif page == "Search & Export":
        st.subheader("Search by AI tags")
        query = st.text_input("Enter a tag or keyword to search")

        if query:
            results = [item for item in st.session_state.images if any(query.lower() in label.lower() for label in item["labels"])]
            
            if not results:
                st.info("No Matches Found!")
            else:
                st.success(f"Found {len(results)} items!")
                cols = st.columns(3)

                for idx, item in enumerate(results):
                    with cols[idx % 3]:
                        st.image(item["image"], use_column_width = True)
                        st.caption(item["name"])
                        
                        with st.expander("Matching Tags"):
                            for _, label, score in item["predictions"]:
                                if query.lower() in label.lower():
                                    st.write(f"**{label}** (match): {score:.2%}")

                                else:
                                    st.write(f"{label}: {score:.2%}")
        st.subheader("Export Gallery Data")

        if st.session_state.images:
            data = []

            for item in st.session_state.images:
                row = {"ID": item["id"], "Name": item["name"], "Timestamp": item["timestamp"]}
                
                for i, (_, label, score) in enumerate(item["predictions"], 1):
                    row[f"Label {i}"] = label
                    row[f"Score {i}"] = f"{score:.2%}"
                
                data.append(row)
            
            df = pd.DataFrame(data)
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index = False)
            
            st.download_button(
                label = "Download CSV",
                data = csv_buffer.getvalue(),
                file_name = "AI_Gallery_Export.csv",
                mime = "text/csv"
            )

            with st.expander("Preview Export Data"):
                st.dataframe(df)
            
        else:
            st.info("Nothing to export yet...")



if __name__ == "__main__":
    main()
