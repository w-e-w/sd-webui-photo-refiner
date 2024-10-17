import gradio as gr
import os
import modules.scripts as scripts
from modules import ui_components, shared, util, paths_internal, scripts_postprocessing
import numpy as np
from PIL import Image, ImageEnhance, ImageChops, ImageFilter
from datetime import datetime


default_output_dir = os.path.join(paths_internal.default_output_dir, 'photo_refiner_outputs')

shared.options_templates.update(shared.options_section(("saving-paths", "Paths for saving"), {
    "photo_refiner_outputs_dir": shared.OptionInfo(util.truncate_path(default_output_dir), 'Output directory for photo refiner images', component_args=shared.hide_dirs),
}))


def apply_effects(img, temperature_value, blur, sharpen, ca, saturation, contrast, brightness, highlights, shadows, film_grain, sepia_filter):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    if temperature_value != 0:
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np[..., 2] += temperature_value * 0.04
        img_np[..., 1] += temperature_value * 0.1 if temperature_value > 0 else img_np[..., 1] - temperature_value * 0.04
        img_np[..., 0] -= temperature_value * 0.04 if temperature_value < 0 else 0
        img_np = np.clip(img_np, 0, 1)
        img = Image.fromarray((img_np * 255).astype(np.uint8))

    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))

    if sharpen > 0:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(sharpen)

    if ca > 0:
        r, g, b = img.split()
        r = ImageChops.offset(r, ca, 0)
        b = ImageChops.offset(b, -ca, 0)
        img = Image.merge('RGB', (r, g, b))

    if saturation != 0:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation / 5.0 + 1)

    if contrast != 0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast / 5.0 + 1)

    if brightness != 0:
        img = Image.fromarray(np.clip(np.array(img) * (brightness / 5.0 + 1), 0, 255).astype(np.uint8))

    if highlights != 0 or shadows != 0:
        img_np = np.array(img).astype(np.float32) / 255.0

        if shadows != 0:
            gamma = 1.0 / (shadows / 5.0 + 1) if shadows > 0 else 1.0 + abs(shadows / 5.0)
            img_np = np.power(img_np, gamma)

        if highlights != 0:
            highlights_scale = highlights / 5.0 + 1
            img_np[img_np > 0.5] = 0.5 + (img_np[img_np > 0.5] - 0.5) * highlights_scale

        img = Image.fromarray(np.clip(img_np * 255, 0, 255).astype(np.uint8))

    if film_grain:
        grain = np.random.normal(0, 1, (img.height, img.width))
        grain = np.clip(grain * 255, 0, 255).astype(np.uint8)
        grain_img = Image.fromarray(grain).convert('L')
        grain_img = grain_img.resize((img.width, img.height), Image.NEAREST)
        grain_img = grain_img.filter(ImageFilter.GaussianBlur(radius=0.7))
        img = Image.blend(img.convert('RGB'), grain_img.convert('RGB'), alpha=0.025)

    if sepia_filter:
        img_np = np.array(img).astype(np.float32) / 255.0
        sepia_effect = np.array(
            [[0.393, 0.769, 0.189],
             [0.349, 0.686, 0.168],
             [0.272, 0.534, 0.131]]
        )
        img_np = img_np.dot(sepia_effect.T)
        img_np = np.clip(img_np, 0, 1)
        img = Image.fromarray((img_np * 255).astype(np.uint8))

    return img


def create_ui():
    with ui_components.InputAccordion(False, label='Photo Refiner') as pr_enabled:
        blur_intensity = gr.Slider(minimum=0, maximum=5, step=0.1, value=0, label="Blur")
        sharpen_intensity = gr.Slider(minimum=0, maximum=10, step=0.1, value=0, label="Sharpening")
        chromatic_aberration = gr.Slider(minimum=0, maximum=3, step=1, value=0, label="Chromatic Aberration")
        saturation_intensity = gr.Slider(minimum=-5, maximum=5, step=0.1, value=0, label="Saturation")
        contrast_intensity = gr.Slider(minimum=-5, maximum=5, step=0.1, value=0, label="Contrast")
        brightness_intensity = gr.Slider(minimum=-5, maximum=5, step=0.1, value=0, label="Brightness")
        highlights_intensity = gr.Slider(minimum=-10, maximum=10, step=0.1, value=0, label="Highlights")
        shadows_intensity = gr.Slider(minimum=-10, maximum=10, step=0.1, value=0, label="Shadows")
        temperature_value = gr.Slider(minimum=-5, maximum=5, step=0.1, value=0, label="Temperature")
        with gr.Row():
            sepia_filter = gr.Checkbox(value=False, label="Sepia Efect")
            film_grain = gr.Checkbox(value=False, label="Filmic Grain")

        reset_button = gr.Button("Reset sliders")

    def reset_sliders():
        return [0] * 10

    def on_reset_button_click():
        return reset_sliders()

    reset_button.click(fn=on_reset_button_click, outputs=[
        blur_intensity,
        sharpen_intensity,
        chromatic_aberration,
        saturation_intensity,
        contrast_intensity,
        brightness_intensity,
        highlights_intensity,
        shadows_intensity,
        temperature_value
    ])

    return {
        'pr_enabled': pr_enabled,
        'temperature_value': temperature_value,
        'blur': blur_intensity,
        'sharpen': sharpen_intensity,
        'ca': chromatic_aberration,
        'saturation': saturation_intensity,
        'contrast': contrast_intensity,
        'brightness': brightness_intensity,
        'highlights': highlights_intensity,
        'shadows': shadows_intensity,
        'film_grain': film_grain,
        'sepia_filter': sepia_filter,
    }


class Script(scripts.Script):

    def title(self, enabled=False):
        return "Photo Refiner 1.2"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return list(create_ui().values())

    def postprocess(self, p, processed, pr_enabled, temperature_value, blur_intensity, sharpen_intensity, chromatic_aberration, saturation_intensity, contrast_intensity, brightness_intensity, highlights_intensity, shadows_intensity, film_grain, sepia_filter, *args):
        if pr_enabled:
            output_dir = shared.opts.photo_refiner_outputs_dir.strip() or default_output_dir
            os.makedirs(output_dir, exist_ok=True)
    
            for i in range(len(processed.images)):
                if isinstance(processed.images[i], np.ndarray):
                    processed_image = Image.fromarray(processed.images[i])
                else:
                    processed_image = processed.images[i]
    
                processed_image = apply_effects(
                    processed_image,
                    temperature_value,
                    blur_intensity,
                    sharpen_intensity,
                    chromatic_aberration,
                    saturation_intensity,
                    contrast_intensity,
                    brightness_intensity,
                    highlights_intensity,
                    shadows_intensity,
                    film_grain,
                    sepia_filter
                )

                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
                file_path = os.path.join(output_dir, f"{timestamp}.png")
                processed_image.save(file_path)
                processed_image.already_saved_as = file_path
                processed.images[i] = processed_image


class PhotoRefinerPP(scripts_postprocessing.ScriptPostprocessing):
    name = "Photo Refiner"
    order = 90000

    def ui(self):
        return create_ui()

    def process(self, pp: scripts_postprocessing.PostprocessedImage, **args):
        if not args.pop('pr_enabled', False):
            return

        pp.image = apply_effects(img=pp.image, **args)
