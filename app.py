import cv2
import numpy as np
import streamlit as st
import imutils

# Function to calculate similarity score
def calculate_similarity_score(image1, image2, threshold_image):
    if image1 is None or image2 is None:
        st.error("Error: Please upload both images.")
        return None

    gray1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2GRAY)
    total_pixels = gray1.shape[0] * gray1.shape[1]

    different_pixels = np.count_nonzero(threshold_image)
    similarity_score = 100 * (1 - (different_pixels / total_pixels))
    return similarity_score

# Function to find differences between images
def find_image_differences(image1, image2):
    gray1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2GRAY)

    diff = cv2.absdiff(gray1, gray2)

    thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    kernel = np.ones((5,5), np.uint8) 
    dilate = cv2.dilate(thresh, kernel, iterations=2) 

    contours = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image1, (x, y), (x+w, y+h), (255,0,0), 2)
            cv2.rectangle(image2, (x, y), (x+w, y+h), (255,0,0), 2)

    return image1, image2, thresh

# Streamlit app
def main():
    st.title("Image Dissimilarity Detection")
    st.sidebar.title("About")
    st.sidebar.info("This app detects differences between two images and calculates their similarity score.")

    # Upload images
    uploaded_image1 = st.sidebar.file_uploader("Upload First Image", type=["jpg", "png", "jpeg"])
    uploaded_image2 = st.sidebar.file_uploader("Upload Second Image", type=["jpg", "png", "jpeg"])

    if uploaded_image1 is not None and uploaded_image2 is not None:
        image1 = cv2.imdecode(np.fromstring(uploaded_image1.read(), np.uint8), 1)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  # Convert to RGB format
        image2 = cv2.imdecode(np.fromstring(uploaded_image2.read(), np.uint8), 1)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)  # Convert to RGB format

        st.subheader("Uploaded Images")
        st.image([image1, image2], caption=["First Image", "Second Image"], width=200)

        if st.button("Find Differences"):
            with st.spinner("Processing..."):
                image1_diff, image2_diff, threshold_image = find_image_differences(image1, image2)
                similarity_score = calculate_similarity_score(image1, image2, threshold_image)

            if similarity_score is not None:
                st.subheader("Differences Detected")
                st.image([image1_diff, image2_diff], caption=["First Image", "Second Image"], width=200)
                st.image(threshold_image, caption="Thresholded Difference", width=200)
                st.success(f"Similarity Score: {similarity_score:.2f}%")

if __name__ == "__main__":
    main()
