import gradio as gr
import PIL.Image
from model_utils import load_model, run_prompt
from cuda_utils import garbage_collect
import time

# Global pipeline variable
pipe = None


def initialize_model():
    """Initialize the FLUX Kontext model pipeline"""
    global pipe
    if pipe is None:
        print("Initializing FLUX Kontext model...")
        pipe = load_model()
        print("Model loaded successfully!")
    return pipe

def process_img2img(input_image: PIL.Image.Image, prompt: str, 
                    guidance_scale: float = 2.5, num_steps: int = 28,
                    progress = gr.Progress(track_tqdm=True)):
    """Process image-to-image inference"""
    if input_image is None:
        return None
    
    if not prompt or prompt.strip() == "":
        return None
    
    try:
        # Initialize model if not already loaded
        pipeline = initialize_model()
        
        # Run inference with custom parameters
        result_image = run_prompt(pipeline, prompt, input_image, guidance_scale, num_steps)
        
        # Clean up GPU memory
        garbage_collect()
        
        return result_image
    
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None

def show_urls():
    """Display the local and shared URLs"""
    global local_url
    global shared_url
    print(f"Local URL: {local_url}")
    print(f"Shared URL: {shared_url}")
    return local_url, shared_url

def move_output_to_input(output_image: PIL.Image.Image):
    """Move the output image to the input"""
    return output_image

def shut_down():
    """Shut down the Gradio interface"""
    global stop
    gr.Warning("Shutting down the server...")
    stop = True
    print("Shutting down the server...")

def create_interface():
    global interface
    """Create the Gradio interface"""
    with gr.Blocks(title="FLUX Kontext - Image to Image") as interface:
        gr.Markdown("# FLUX Kontext Image-to-Image Generator")
        gr.Markdown("Upload an image and provide a prompt to generate a new image using FLUX Kontext.")
        
        with gr.Row():
            with gr.Column(scale=2):
                img2img_input = gr.Image(label="Input Image", type="pil")
            with gr.Column(scale=1):
                img2img_prompt = gr.Textbox(label="Prompt", lines=5, max_lines=10, placeholder="Enter your prompt here...", value="")
                img2img_generate_button = gr.Button("Generate")
        with gr.Row():
            # Advanced settings in accordion
            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    guidance_scale_slider = gr.Slider(
                        minimum=1.0, maximum=10.0, value=2.5, step=0.1,
                        label="Guidance Scale",
                        info="Higher values follow the prompt more closely (1.0-10.0)"
                    )
                    num_steps_slider = gr.Slider(
                        minimum=1, maximum=50, value=28, step=1,
                        label="Number of Steps",
                        info="More steps = better quality but slower (1-50)"
                    )
        
        with gr.Row():
            with gr.Column(scale=7):
                img2img_output = gr.Image(label="Output Image", type="pil", interactive=False)
            with gr.Column(scale=1):
                move_up_button = gr.Button("↑ Move to Input", variant="secondary")
        with gr.Row():
            with gr.Column(scale=1):
                show_urls_button = gr.Button("Show URLs", variant="primary")
            with gr.Column(scale=3):
                local_url_text = gr.Textbox(label="Local URL", interactive=False)
            with gr.Column(scale=3):
                shared_url_text = gr.Textbox(label="Shared URL", interactive=False)
        with gr.Row():
            shutdown_button = gr.Button("Shutdown", variant="stop", elem_id="shutdown-button")  
        shutdown_button.click(shut_down)

        show_urls_button.click(show_urls, outputs=[local_url_text, shared_url_text])

        img2img_generate_button.click(
            process_img2img, 
            [img2img_input, img2img_prompt, guidance_scale_slider, num_steps_slider], 
            [img2img_output]
        )
        move_up_button.click(move_output_to_input, [img2img_output], [img2img_input])
    
    return interface

if __name__ == "__main__":
    global local_url
    global shared_url
    global stop
    stop = False
    local_url = ""
    shared_url = ""
    # Create and launch the interface
    interface = create_interface()
    interface.queue()
    _, local_url, shared_url = interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        prevent_thread_lock=True
    )
    print(f"Local URL: {local_url}")
    print(f"Shared URL: {shared_url}")
    try:
        while not stop:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down...")
