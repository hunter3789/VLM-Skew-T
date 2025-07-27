import os

# Set environment variables for offline mode
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_OFFLINE"] = "1"

#print(os.environ.get("TRANSFORMERS_OFFLINE"))  # Should print 1
#print(os.environ.get("HF_HUB_OFFLINE"))         # Should print 1

from pathlib import Path
import json
from base_vlm import BaseVLM
from data import VQADataset
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments
from transformers import AutoProcessor, AutoTokenizer, AutoModelForVision2Seq
from PIL import Image
from transformers.image_utils import load_image
import torch
import torch.nn.functional as F
import numpy as np
import scipy.ndimage as ndi         # or: import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.cm import ScalarMappable

DATA_DIR = Path(__file__).parent.parent / "data"


def get_kw_positions(tokenizer, full_ids, keyword, img_token_idx):
    """Return *all* starting indices where `keyword` occurs (handles sub-token splits)."""
    sub = tokenizer(keyword, add_special_tokens=False)["input_ids"]
    return [(i,i+len(sub)) for i in range(len(full_ids) - len(sub) + 1)
            if full_ids[i:i + len(sub)] == sub and i > img_token_idx]

def attention_map(ckpt_path: str):
    vlm = BaseVLM()

    # Load the model with LoRA adapters
    from peft import PeftModel
    import random

    #vlm.model = PeftModel.from_pretrained(vlm.model, ckpt_path, local_files_only=True).to(vlm.device)
    processor = vlm.processor
    tokenizer = processor.tokenizer

    vlm.model.eval()

    # Use a sample image from the internet
    current_dir = Path(__file__).parent
    #image_path = str((current_dir / "../data/valid_imgs/skew_184_2024090100.png").resolve())
    #image_path = str((current_dir / "../data/valid_imgs/skew_184_2024062000.png").resolve())
    #image_path = str((current_dir / "../data/valid_imgs/skew_108_2024061921.png").resolve())
    #image_path = str((current_dir / "../data/valid_imgs/skew_105_2024072418.png").resolve())
    image_path = str((current_dir / "../data/valid_imgs/skew_105_2024061121.png").resolve())
    image_paths = [image_path]
    images = [[load_image(img_path)] for img_path in image_paths]

    message = [
        #{
        #    "role": "system",
        #    "content": [
        #        {"type": "text", "text": "You are a weather forecaster analyzing atmospheric soundings shown in Skew-T log-P diagrams.\n\n- Lower layer: 1000–850 hPa\n- Mid layer: 850–500 hPa\n- Upper layer: 500–250 hPa\n\nDiagram legend:\n- Red line: temperature\n- Green line: dew point temperature\n- Shaded blue area: CAPE (Convective Available Potential Energy)\n- Shaded yellow area: CIN (Convective Inhibition)\n\nMeteorological interpretation tips:\n- When the red and green lines are close, the atmosphere is moist.\n- The LFC (Level of Free Convection) is the lowest point of the blue area.\n- The EL (Equilibrium Level) is the highest point of the blue area.\n- Wind barbs are displayed on the right side. If they rotate clockwise with height, it indicates veering winds; if counterclockwise, it indicates backing.\n\nPlease describe the atmospheric profile based on the provided Skew-T log-P diagram. Reason carefully, and conclude with a precipitation probability category: Low, Moderate, High, or Very High."}
        #    ]
        #},
        {
        #    "role": "user",
            "content": [
               {"type": "image"},  # Correct type to insert image token
        #      {"type": "text", "text": prompt},
               {"type": "text", "text": "You are a weather forecaster analyzing atmospheric soundings shown in a Skew-T log-P diagram\nThe diagram uses a logarithmic vertical pressure axis (hPa), so pressure layers are not evenly spaced. Use the following visual anchors:\nLower layer (1000-850 hPa): This is located in the bottom quarter of the diagram, close to the surface. It represents the boundary layer where surface temperature, dew point, and CIN typically appear.\nMid layer (850-500 hPa): Appears in the second quarter from the bottom of the plot. This region often contains most of the CAPE and developing updrafts.\nUpper layer (500-250 hPa): This is around the middle third of the diagram, despite covering less pressure range. This layer includes the top of convection (EL), cirrus clouds, and upper-level wind shear.\nUse the following visual references:\nRed line: temperature profile\nGreen line: dew point temperature\nBlue shaded area: CAPE (Convective Available Potential Energy)\nYellow shaded area: CIN (Convective Inhibition)\nWind barbs: on the right-hand side, changing with height\nKey interpretation rules:\nWhere the red and green lines are close, the layer is moist; nearly overlap indicates saturated; far apart implies dryness.\nThe LFC is the bottom of the blue area; the EL is the top of the blue area.\nClockwise turning wind barbs with height suggest veering (warm air advection); counterclockwise suggests backing."}
            ]
        }
    ]   
    prompts = [processor.apply_chat_template(message, add_generation_prompt=True)]

    # === 3. Preprocess ===
    inputs = processor(text=prompts, images=images, return_tensors="pt")
    inputs = {k: v.to(vlm.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = vlm.model(**inputs, output_attentions=True, return_dict=True)

    # === 4. Decode token sequence ===
    input_ids = inputs["input_ids"][0]
    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)
    img_token_idx = next(i for i, tok in enumerate(tokens) if tok == '<image>')
    #print(tokens)
    #print(img_token_idx)

    # === 5. Get index of target word ===
    target_word = [" layer is moist", " far apart implies dryness", "Upper layer (500-250 hPa):", "Mid layer (850-500 hPa):", "Lower layer (1000-850 hPa):", "Wind barbs: on the right", "Blue shaded area: CAPE", "Yellow shaded area: CIN"]
    attn_map_list = []

    for t in target_word:
        token_idx = get_kw_positions(tokenizer, inputs["input_ids"][0].tolist(), t, img_token_idx)[0]
        print(token_idx)

        # === 6. Get self-attention from decoder ===
        attn_stack = torch.stack([a[0] for a in outputs.attentions], dim=0)  # [layers, heads, seq, seq]
        attn_avg = attn_stack.mean(dim=(0, 1))  # average over layers & heads → [seq, seq]
        #attn_avg = attn_stack.amax(dim=(0, 1))  # average over layers & heads → [seq, seq]
        #self_attn = outputs.attentions[-1][0]  # last layer, batch=0 → [heads, seq, seq]
        #attn_avg = self_attn.mean(dim=0)       # → [seq, seq]

        # === 7. Define visual token range (adjust if needed) ===
        num_visual_tokens = outputs.image_hidden_states[0].shape[-2]
        visual_attn = attn_avg[token_idx[0]:token_idx[1], img_token_idx:img_token_idx+num_visual_tokens].mean(0)  # [num_visual_tokens]

        npatch = int(np.sqrt(num_visual_tokens))

        # === 8. Reshape & Upsample ===
        attn_map = visual_attn.reshape(npatch, npatch)  
        print(attn_map)
        #attn_map[0,:]  = attn_map[1,:]
        #attn_map[-1,:] = attn_map[-2,:]
        #attn_map[:,0]  = attn_map[:,1]
        #attn_map[:,-1] = attn_map[:,-2]

        #attn_map_up = F.interpolate(
        #    attn_map.unsqueeze(0).unsqueeze(0),
        #    size=inputs["pixel_values"].shape[-2:],  # [H, W]
        #    mode='nearest',
        #)

        attn_map_list.append(attn_map)

    # === 9. Visualize and Save ===
    def save_attention_overlay(image_tensor, attn_map_list, token_label_list, out_path):
        img = image_tensor.squeeze().permute(1, 2, 0).to(torch.float32).cpu().numpy()

        img_w, img_h = img.shape[:2]

        nrow, ncol = 3, 3
        fig, axs = plt.subplots(nrow, ncol, figsize=(3*ncol+1, 3*nrow))
        axs[0][0].imshow(img)
        axs[0][0].axis('off')

        cmap = plt.get_cmap("jet")

        for index, (attn_map, token_label) in enumerate(zip(attn_map_list, token_label_list)):
            ax = axs[(index+1) // ncol][(index+1) % ncol]
            ax.imshow(img)

            heatmap = attn_map.squeeze().to(torch.float32).cpu().numpy()
            heatmap = ndi.gaussian_filter(heatmap, sigma=1.0)   # <-- float-preserving
            heatmap = (heatmap - heatmap.min()) / (np.ptp(heatmap) + 1e-8)  # normalize 0–1

            grid_w, grid_h = heatmap.shape
            cell_w, cell_h = img_w / grid_w, img_h / grid_h

            # Draw colored rectangles
            for i in range(grid_h):        # vertical = y
                for j in range(grid_w):    # horizontal = x
                    val = heatmap[i, j]
                    color = cmap(val)
                    rect = patches.Rectangle(
                        (j * cell_w, i * cell_h),       # x, y
                        width  = cell_w,
                        height = cell_h,
                        linewidth = 0,
                        edgecolor = None,
                        facecolor = color,
                        alpha     = 0.45,
                    )
                    ax.add_patch(rect)

            ax.set_title(f"{token_label}", fontsize=14)
            ax.axis('off')
            #plt.tight_layout()

        # Add colorbar
        # 1. Define normalization (match your heat range)
        vmin = 0.0
        vmax = 1.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # 2. Create a dummy ScalarMappable
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # required for compatibility

        cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_label("Attention Weight (Normalized)", fontsize=12)  # Label font size
        cbar.ax.tick_params(labelsize=12)                # Tick label font size

        plt.savefig(out_path)
        plt.close()

    save_attention_overlay(inputs["pixel_values"], attn_map_list, target_word, "attention_map.png")
    print("Saved attention map")


def test_model(ckpt_path: str):
    import random

    testset = VQADataset("valid_vlm_diagram_QA")
    vlm = BaseVLM()

    # Load the model with LoRA adapters
    from peft import PeftModel

    vlm.model = PeftModel.from_pretrained(vlm.model, ckpt_path, local_files_only=True).to(vlm.device)

    d = random.sample(testset.qa_pairs, 10)

    #print(d["question"])
    #image_path = str(d["image_path"])
    #print(image_path)
    image_path = [str(o["image"]) for o in d]
    prompts = [o["system"] for o in d]
    questions = [o["user"] for o in d]
    answers = [o["response"] for o in d]

    responses = vlm.answer(image_path, prompts, questions, temperature = 0, use_images=True)

    for q, r, a in zip(questions, responses, answers):
        print(f"Q: {q}")
        print(f"\nR: {r}")
        print(f"A: {a}\n\n")

    #answer = vlm.answer([image_path], [d["system"]], [d["question"]], temperature = 0, use_images=True)
    #print("")
    #print(answer)

    #print("")
    #print(answers)


if __name__ == "__main__":
    #attention_map("attention")
    attention_map("vlm_sft_QA_2.2B")
    #test_model("./vlm_sft_QA_2.2B")