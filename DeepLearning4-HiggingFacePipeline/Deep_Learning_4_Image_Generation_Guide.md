# Deep Learning 4: Image Generation with Diffusion Models

## Complete Step-by-Step Instructions

### Overview

This guide provides comprehensive instructions for completing the Deep Learning 4 assignment on image generation using diffusion models from Hugging Face. You'll learn to generate high-quality images using state-of-the-art diffusion models.

---

## Prerequisites

### Required Libraries

```bash
pip install diffusers transformers torch torchvision accelerate
pip install matplotlib pillow numpy
```

### System Requirements

- **GPU Recommended**: CUDA-compatible GPU with at least 8GB VRAM
- **CPU Alternative**: Will work on CPU but much slower
- **RAM**: At least 16GB system RAM recommended

---

## Step 1: Setup and Environment Preparation

### 1.1 Import Required Libraries

```python
# Core libraries
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Optional: For advanced features
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
```

### 1.2 Device Configuration

```python
# Check and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Enable memory optimizations
if torch.cuda.is_available():
    torch.backends.cuda.enable_math_sdp(True)
    print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

## Step 2: Choose and Load a Diffusion Model

### 2.1 Model Selection Options

**Option A: Stable Diffusion 2.1 (Recommended for beginners)**

```python
model_id = "stabilityai/stable-diffusion-2-1"
# Pros: Good balance of quality and speed, well-documented
# Cons: Slightly older architecture
```

**Option B: Stable Diffusion XL (Highest quality)**

```python
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
# Pros: Highest quality, better text understanding
# Cons: More resource intensive, slower generation
```

**Option C: Stable Diffusion 1.5 (Fastest)**

```python
model_id = "runwayml/stable-diffusion-v1-5"
# Pros: Fastest generation, good for experimentation
# Cons: Lower quality than newer models
```

### 2.2 Load the Pipeline

```python
print(f"Loading model: {model_id}")

# Load the pipeline with optimizations
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    use_safetensors=True,
    safety_checker=None,  # Disable for faster loading (optional)
    requires_safety_checker=False
)

# Use DPMSolver for faster inference
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Move to device
pipe = pipe.to(device)

# Enable memory optimizations
pipe.enable_attention_slicing()
if device == "cuda":
    pipe.enable_model_cpu_offload()

print("✅ Model loaded successfully!")
```

---

## Step 3: Generate Images with Different Prompts

### 3.1 Define Your Prompts

```python
# Create diverse prompts for testing
prompts = [
    "A beautiful sunset over a mountain landscape, digital art, highly detailed, 4k",
    "A futuristic city with flying cars, cyberpunk style, neon lights, night scene",
    "A cute cat sitting on a windowsill, watercolor painting, soft lighting",
    "Abstract art with vibrant colors and geometric shapes, modern art style",
    "A medieval knight in shining armor, fantasy art style, detailed armor"
]

# Alternative: Create themed prompt sets
landscape_prompts = [
    "Serene mountain lake at sunrise, photorealistic",
    "Dense forest with sunlight filtering through trees",
    "Desert landscape with sand dunes, golden hour lighting"
]

portrait_prompts = [
    "Professional headshot of a businesswoman, studio lighting",
    "Elderly man with kind eyes, black and white photography",
    "Young artist in their studio, natural lighting"
]
```

### 3.2 Set Generation Parameters

```python
# Standard generation parameters
generation_params = {
    "num_inference_steps": 20,        # 20-50 recommended (higher = better quality, slower)
    "guidance_scale": 7.5,           # 7.5-15 (higher = more prompt adherence)
    "width": 512,                    # Image width (512, 768, 1024)
    "height": 512,                   # Image height
    "num_images_per_prompt": 1,      # Number of images per prompt
    "seed": None                     # Set seed for reproducible results
}

# High quality parameters (slower)
high_quality_params = {
    "num_inference_steps": 50,
    "guidance_scale": 10.0,
    "width": 768,
    "height": 768,
    "num_images_per_prompt": 1
}
```

### 3.3 Generate Images

```python
# Generate images for each prompt
generated_images = []
generation_info = []

for i, prompt in enumerate(prompts):
    print(f"🎨 Generating image {i+1}/{len(prompts)}: {prompt[:50]}...")

    try:
        with torch.autocast(device):
            result = pipe(
                prompt=prompt,
                **generation_params
            )

        image = result.images[0]
        generated_images.append(image)
        generation_info.append({
            'prompt': prompt,
            'seed': result.seed,
            'generation_time': 'N/A'  # You can add timing if needed
        })

        print(f"✅ Generated image {i+1}")

    except Exception as e:
        print(f"❌ Error generating image {i+1}: {str(e)}")
        continue

print(f"🎉 Successfully generated {len(generated_images)} images!")
```

---

## Step 4: Display and Analyze Generated Images

### 4.1 Create Image Grid

```python
# Create a figure to display all images
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (image, prompt) in enumerate(zip(generated_images, prompts)):
    axes[i].imshow(image)
    axes[i].set_title(f"Image {i+1}: {prompt[:40]}...", fontsize=10, wrap=True)
    axes[i].axis('off')

# Hide unused subplots
for i in range(len(generated_images), len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

### 4.2 Individual Image Display

```python
# Display images individually with full details
for i, (image, info) in enumerate(zip(generated_images, generation_info)):
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(f"Generated Image {i+1}", fontsize=14)
    plt.axis('off')

    # Add prompt as text
    plt.figtext(0.5, 0.02, f"Prompt: {info['prompt']}",
                ha='center', fontsize=10, wrap=True)
    plt.show()
```

### 4.3 Save Generated Images

```python
# Save all generated images
import os
os.makedirs("generated_images", exist_ok=True)

for i, image in enumerate(generated_images):
    filename = f"generated_images/image_{i+1}.png"
    image.save(filename)
    print(f"💾 Saved: {filename}")
```

---

## Step 5: Experiment with Different Parameters

### 5.1 Test Different Guidance Scales

```python
# Test how guidance scale affects generation
test_prompt = "A serene lake with mountains in the background, photorealistic"
guidance_scales = [5.0, 7.5, 10.0, 15.0]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for i, guidance in enumerate(guidance_scales):
    print(f"Testing guidance scale: {guidance}")

    image = pipe(
        prompt=test_prompt,
        num_inference_steps=20,
        guidance_scale=guidance,
        width=512,
        height=512
    ).images[0]

    axes[i].imshow(image)
    axes[i].set_title(f"Guidance Scale: {guidance}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

### 5.2 Test Different Inference Steps

```python
# Test how inference steps affect quality
inference_steps = [10, 20, 30, 50]
test_prompt = "A detailed portrait of a wise old wizard, fantasy art"

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for i, steps in enumerate(inference_steps):
    print(f"Testing inference steps: {steps}")

    image = pipe(
        prompt=test_prompt,
        num_inference_steps=steps,
        guidance_scale=7.5,
        width=512,
        height=512
    ).images[0]

    axes[i].imshow(image)
    axes[i].set_title(f"Steps: {steps}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

---

## Step 6: Advanced Techniques

### 6.1 Using Negative Prompts

```python
# Negative prompts help avoid unwanted elements
prompt = "A beautiful landscape with mountains and trees"
negative_prompt = "blurry, low quality, distorted, ugly, deformed, oversaturated"

# Generate with and without negative prompts
image_with_negative = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=25,
    guidance_scale=8.0,
    width=512,
    height=512
).images[0]

image_without_negative = pipe(
    prompt=prompt,
    num_inference_steps=25,
    guidance_scale=8.0,
    width=512,
    height=512
).images[0]

# Compare results
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image_without_negative)
axes[0].set_title("Without Negative Prompt")
axes[0].axis('off')

axes[1].imshow(image_with_negative)
axes[1].set_title("With Negative Prompt")
axes[1].axis('off')

plt.tight_layout()
plt.show()
```

### 6.2 Seed Control for Reproducibility

```python
# Use seeds for reproducible results
seed = 42
prompt = "A futuristic robot in a cyberpunk city"

# Generate multiple images with same seed
images_same_seed = []
for i in range(3):
    image = pipe(
        prompt=prompt,
        seed=seed,
        num_inference_steps=20,
        guidance_scale=7.5,
        width=512,
        height=512
    ).images[0]
    images_same_seed.append(image)

# Display results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, image in enumerate(images_same_seed):
    axes[i].imshow(image)
    axes[i].set_title(f"Seed: {seed} - Run {i+1}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

---

## Step 7: Discussion and Analysis Template

### 7.1 Model Selection Discussion

```python
print("=" * 80)
print("ASSIGNMENT DISCUSSION: IMAGE GENERATION WITH DIFFUSION MODELS")
print("=" * 80)

print("\n1. MODEL CHOSEN AND REASONING:")
print("   Model: stabilityai/stable-diffusion-2-1")
print("   Reasoning:")
print("   - Good balance between quality and generation speed")
print("   - Well-documented and widely used")
print("   - Compatible with most hardware configurations")
print("   - Alternative models considered:")
print("     * SDXL: Higher quality but more resource intensive")
print("     * SD 1.5: Faster but lower quality")
```

### 7.2 Prompts and Parameters Analysis

```python
print("\n2. PROMPTS AND PARAMETERS USED:")
print("   Prompts tested:")
for i, prompt in enumerate(prompts, 1):
    print(f"   {i}. {prompt}")

print("\n   Parameters used:")
print("   - Inference Steps: 20 (balance of quality vs speed)")
print("   - Guidance Scale: 7.5 (good prompt adherence)")
print("   - Image Size: 512x512 (standard resolution)")
print("   - Scheduler: DPMSolverMultistepScheduler (faster inference)")
```

### 7.3 Quality Observations

```python
print("\n3. OBSERVATIONS ABOUT GENERATED IMAGES:")
print("   Quality Assessment:")
print("   - Overall quality: High, with good detail and coherence")
print("   - Style adherence: Model follows artistic style instructions well")
print("   - Consistency: Some variation between runs (expected behavior)")
print("   - Generation time: ~15-30 seconds per image on GPU")
print("   - Memory usage: ~6-8GB VRAM for 512x512 images")

print("\n   Strengths observed:")
print("   - Excellent at landscape and nature scenes")
print("   - Good understanding of artistic styles")
print("   - Handles complex prompts well")
print("   - Consistent color and lighting")

print("\n   Areas for improvement:")
print("   - Sometimes struggles with human faces")
print("   - Occasional artifacts in fine details")
print("   - May need multiple attempts for perfect results")
```

### 7.4 Challenges and Solutions

```python
print("\n4. CHALLENGES ENCOUNTERED AND SOLUTIONS:")
print("   Technical Challenges:")
print("   - Memory usage: Large models require significant VRAM")
print("     Solution: Used attention slicing and model offloading")
print("   - Generation speed: Can be slow without optimization")
print("     Solution: Used DPMSolver scheduler and reduced inference steps")
print("   - Quality consistency: Some generations vary in quality")
print("     Solution: Experimented with different parameters and seeds")

print("\n   Prompt Engineering Challenges:")
print("   - Finding the right balance of detail vs clarity")
print("   - Avoiding overly complex prompts that confuse the model")
print("   - Using negative prompts effectively")
print("   - Solution: Iterative testing and refinement")
```

### 7.5 Interesting Findings

```python
print("\n5. INTERESTING FINDINGS AND INSIGHTS:")
print("   Technical Insights:")
print("   - Negative prompts significantly improve image quality")
print("   - Guidance scale has major impact on prompt adherence")
print("   - Different models excel at different content types")
print("   - Seed control enables reproducible results")

print("\n   Creative Insights:")
print("   - Model shows strong understanding of artistic styles")
print("   - Good at combining multiple concepts in prompts")
print("   - Effective at generating atmospheric scenes")
print("   - Can create convincing photorealistic images")

print("\n   Practical Applications:")
print("   - Useful for concept art and ideation")
print("   - Good for creating reference images")
print("   - Effective for exploring creative directions")
print("   - Valuable for educational and research purposes")
```

---

## Step 8: Advanced Experiments (Optional)

### 8.1 Model Comparison

```python
# Compare different models with same prompt
models_to_compare = [
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-2-1"
]

comparison_prompt = "A majestic dragon flying over a medieval castle"

# Load and test each model
for model_id in models_to_compare:
    print(f"Testing model: {model_id}")
    # Load model and generate image
    # (Implementation similar to previous steps)
```

### 8.2 Image-to-Image Generation

```python
# Load image-to-image pipeline
from diffusers import StableDiffusionImg2ImgPipeline

img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

# Load a base image
base_image = Image.open("your_image.jpg")

# Generate variations
result = img2img_pipe(
    prompt="Transform this into a cyberpunk style",
    image=base_image,
    strength=0.7,  # How much to change the original
    num_inference_steps=20
)
```

---

## Step 9: Memory Management and Cleanup

### 9.1 Clean Up Resources

```python
# Clean up memory after generation
del pipe
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("GPU memory cleared")

# Optional: Force garbage collection
import gc
gc.collect()
print("✅ Memory cleaned up successfully!")
```

---

## Troubleshooting Guide

### Common Issues and Solutions

**Issue: Out of Memory Error**

```python
# Solutions:
# 1. Reduce image size
generation_params["width"] = 256
generation_params["height"] = 256

# 2. Use CPU offloading
pipe.enable_model_cpu_offload()

# 3. Reduce batch size
generation_params["num_images_per_prompt"] = 1

# 4. Use attention slicing
pipe.enable_attention_slicing()
```

**Issue: Slow Generation**

```python
# Solutions:
# 1. Reduce inference steps
generation_params["num_inference_steps"] = 10

# 2. Use faster scheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# 3. Use smaller model
model_id = "runwayml/stable-diffusion-v1-5"
```

**Issue: Poor Quality Images**

```python
# Solutions:
# 1. Increase inference steps
generation_params["num_inference_steps"] = 50

# 2. Adjust guidance scale
generation_params["guidance_scale"] = 10.0

# 3. Improve prompts
# - Be more specific
# - Add quality keywords: "high quality", "detailed", "4k"
# - Use negative prompts
```

**Issue: Inconsistent Results**

```python
# Solutions:
# 1. Use fixed seeds
generation_params["seed"] = 42

# 2. Use deterministic settings
torch.manual_seed(42)

# 3. Increase inference steps for stability
generation_params["num_inference_steps"] = 30
```

---

## Final Checklist

Before submitting your assignment, ensure you have:

- [ ] Successfully loaded a diffusion model
- [ ] Generated at least 3-5 different images
- [ ] Displayed images with proper titles and labels
- [ ] Experimented with different parameters
- [ ] Included discussion covering all required points:
  - [ ] Model choice and reasoning
  - [ ] Prompts and parameters used
  - [ ] Quality observations
  - [ ] Challenges encountered
  - [ ] Interesting findings
- [ ] Saved generated images
- [ ] Cleaned up memory properly

---

## Additional Resources

- [Hugging Face Diffusers Documentation](https://huggingface.co/docs/diffusers/)
- [Stable Diffusion Models on Hugging Face](https://huggingface.co/stabilityai)
- [Prompt Engineering Guide](https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline)
- [Diffusion Models Explained](https://huggingface.co/blog/annotated-diffusion)

---

## Tips for Success

1. **Start Simple**: Begin with basic prompts and gradually add complexity
2. **Experiment Systematically**: Test one parameter at a time
3. **Document Everything**: Keep track of what works and what doesn't
4. **Use Negative Prompts**: They can significantly improve quality
5. **Monitor Resources**: Watch memory usage and generation time
6. **Save Your Work**: Always save generated images and parameters
7. **Be Patient**: High-quality generation takes time
8. **Iterate**: Don't expect perfect results on the first try

Good luck with your assignment! 🎨✨
