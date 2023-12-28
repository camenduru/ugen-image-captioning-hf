import gradio as gr
from uform import gen_model
from PIL import Image
import torch

# Load the model and processor
model = gen_model.VLMForCausalLM.from_pretrained("unum-cloud/uform-gen")
processor = gen_model.VLMProcessor.from_pretrained("unum-cloud/uform-gen")

def generate_caption(image, prompt):
    # Process the image and the prompt
    inputs = processor(texts=[prompt], images=[image], return_tensors="pt")

    # Generate the output
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            use_cache=True,
            max_new_tokens=128,
            eos_token_id=32001,
            pad_token_id=processor.tokenizer.pad_token_id
        )

    prompt_len = inputs["input_ids"].shape[1]
    decoded_text = processor.batch_decode(output[:, prompt_len:])[0]

    return decoded_text

# Define the Gradio interface
iface = gr.Interface(
    fn=generate_caption,
    inputs=[gr.Image(type="pil", label="Upload Image"), gr.Textbox(label="Prompt")],
    outputs=gr.Textbox(label="Generated Caption"),
)

# Launch the interface
iface.launch()
