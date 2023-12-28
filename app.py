import gradio as gr
from uform import gen_model
from PIL import Image

# Load the model and processor
model = gen_model.VLMForCausalLM.from_pretrained("unum-cloud/uform-gen")
processor = gen_model.VLMProcessor.from_pretrained("unum-cloud/uform-gen")

def generate_caption(image, prompt):
    # Process the image and the prompt
    inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True)

    # Generate the output
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption

# Load the demo image
demo_image = Image.open("jungle-glass.png")

# Define the Gradio interface
iface = gr.Interface(
    fn=generate_caption,
    inputs=[gr.Image(type="pil", label="Upload Image", default=demo_image), gr.Textbox(label="Prompt")],
    outputs=gr.Textbox(label="Generated Caption"),
)

# Launch the interface
iface.launch()
