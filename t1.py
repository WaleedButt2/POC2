import gradio as gr
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageOps
import numpy as np

# Create an inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="hdqxK3b7M92jruMhYWdn"
)

def load_image_correct_orientation(image):
    return ImageOps.exif_transpose(image)

def process_image(image):
    # Check if image is a NumPy array and convert to PIL Image if true
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Save the PIL image to a temporary path to use with the inference client
    image_path = '/tmp/uploaded_image.jpg'
    image.save(image_path)

    # Perform inference
    result = CLIENT.infer(image_path, model_id="albums/1")

    # Load and correct image orientation
    original_image = load_image_correct_orientation(image)
    draw = ImageDraw.Draw(original_image, 'RGBA')

    # Assuming 'result' is your segmentation output dictionary
    for prediction in result['predictions']:
        points = prediction['points']
        # Create a polygon from points and draw it on the image
        xy = [(point['x'], point['y']) for point in points]
        draw.polygon(xy, fill=(255, 0, 0, 128))  # Red color with half transparency

    return original_image

# Gradio interface definition
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(label="Upload Image"),
    outputs=gr.Image(),
    title="Image Segmentation Viewer",
    description="Upload an image to view its segmentation annotations."
)

# Run the Gradio app with sharing enabled
iface.launch(share=True)
