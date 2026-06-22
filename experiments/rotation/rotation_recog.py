import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.infer import InferenceModel
import time
import argparse
import logging
import random
import pandas as pd
import random
import re
from google.genai.errors import ServerError
from tqdm import tqdm

def safe_infer(model, infer_args, max_retries=5):
    """
    Run model inference with automatic retry on 503 UNAVAILABLE errors.
    Implements exponential backoff to handle overloaded Gemini servers.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return model.infer(infer_args)
        except ServerError as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                wait_time = (2 ** attempt) + random.random() * 2  # jitter
                print(f"[WARN] Gemini overloaded (attempt {attempt}/{max_retries}). Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                raise
    raise RuntimeError(f"Model unavailable after {max_retries} retries. Aborting inference.")

def parse_response(response_text):
    """
    Parse response text to extract Yes/No answer from curly brackets.
    Returns 'yes' or 'no' (lowercase) if found, None otherwise.
    """
    if not response_text:
        return None
    # Look for {Yes} or {No} in curly brackets (case insensitive)
    match = re.search(r'\{([^}]+)\}', str(response_text), re.IGNORECASE)
    if match:
        answer = match.group(1).strip().lower()
        if 'yes' in answer:
            return 'yes'
        elif 'no' in answer:
            return 'no'
    return None

def calculate_metrics(responses):
    """
    Calculate accuracy and recall from responses dictionary.
    Recall is more faithful for rotation detection - measures ability to detect actual rotations.
    Returns (correct, total, accuracy_percentage, recall_percentage, tp, fp, tn, fn)
    """
    if not responses['response'] or not responses['ground_truth']:
        return 0, 0, 0.0, 0.0, 0, 0, 0, 0
    
    correct = 0
    total = len(responses['response'])
    tp = 0  # True Positive: ground_truth='yes', response='yes'
    fp = 0  # False Positive: ground_truth='no', response='yes'
    tn = 0  # True Negative: ground_truth='no', response='no'
    fn = 0  # False Negative: ground_truth='yes', response='no'
    
    for response, ground_truth in zip(responses['response'], responses['ground_truth']):
        parsed = parse_response(response)
        gt_lower = ground_truth.lower()
        
        if parsed and parsed == gt_lower:
            correct += 1
            
        # Calculate confusion matrix components
        if parsed == 'yes' and gt_lower == 'yes':
            tp += 1
        elif parsed == 'yes' and gt_lower == 'no':
            fp += 1
        elif parsed == 'no' and gt_lower == 'no':
            tn += 1
        elif parsed == 'no' and gt_lower == 'yes':
            fn += 1
    
    accuracy = (correct / total * 100) if total > 0 else 0.0
    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
    
    return correct, total, accuracy, recall, tp, fp, tn, fn

def rotation_recog(script, 
                 prompt_name, 
                 text_prompt, 
                 gemini_start_idx=0, 
                 identity=False, 
                 flip_axis=None,
                 boxed=False,
                 model_config={"model_name": "gemini-2.5-pro"},
                 log_path=None,
                 data_dir=None,
                 output_dir=None
                ):
    start_time = time.time()
    FORMAT = '%(asctime)s %(message)s'
    logger = logging.getLogger(__name__)
    if log_path is not None:
        logging.basicConfig(filename=log_path, level=logging.INFO, format=FORMAT)
    
    model_name = "qwen2.5_vl" if "qwen" in model_config.get("model_name", "").lower() else "gemini2.5_pro"
    model = InferenceModel(model_config["model_name"])
    logger.info(f'{script} Finished Model Initialization')

    OMNIGLOT_FOLDER = os.path.join(data_dir, "images_original", script)
    if type(flip_axis) == str:

        if flip_axis.lower() == "y":
            TRANSFORMED_IMAGE_FOLDER = "images_flipped_horizontal"
        elif flip_axis.lower() == "x":
            raise ValueError("Have not created images_flipped_vertical directory yet")
    elif boxed:
        TRANSFORMED_IMAGE_FOLDER = "images_rotated_box"
    else:
        TRANSFORMED_IMAGE_FOLDER = "images_rotated"
    logger.info(f"flip axis: {flip_axis}")
    logger.info(f"flip axis type: {type(flip_axis)}")
    logger.info(f"transformed image folder: {TRANSFORMED_IMAGE_FOLDER}")
    TRANSFORMED_OMNIGLOT_FOLDER = os.path.join(data_dir, TRANSFORMED_IMAGE_FOLDER, script)

    request_count = 0
    responses = {"script": [], "character_id": [], "image_id": [], "angle": [], "response": [], 'ground_truth': []}

    char_paths = [os.path.join(OMNIGLOT_FOLDER, char_dir) for char_dir in sorted(os.listdir(OMNIGLOT_FOLDER))]
    transformed_char_paths = [os.path.join(TRANSFORMED_OMNIGLOT_FOLDER, char_dir) for char_dir in sorted(os.listdir(TRANSFORMED_OMNIGLOT_FOLDER))]
    
    char_range = range(gemini_start_idx, len(char_paths))
    char_pbar = tqdm(char_range, desc="Characters")
    for idx in char_pbar:
        char_path = char_paths[idx]
        transformed_char_path = transformed_char_paths[idx]
        char_id = os.path.basename(transformed_char_path)

        image_paths = [os.path.join(char_path, img) for img in sorted(os.listdir(char_path))]
        transformed_image_paths = [os.path.join(transformed_char_path, img_dir) for img_dir in sorted(os.listdir(transformed_char_path))]

        img_pbar = tqdm(zip(image_paths, transformed_image_paths), total=len(image_paths), desc="Images", leave=False)
        for original_image, transformed_image_dir in img_pbar:
            img_id = os.path.basename(transformed_image_dir)
            
            if boxed:
                zero_file = os.listdir(os.path.join(transformed_image_dir, "Angle_0"))[0]
                original_image = os.path.join(transformed_image_dir, "Angle_0", zero_file)

            random_image = original_image
            while random_image == original_image:
                random_char = random.choice(char_paths)
                random_image = os.path.join(random_char, random.choice(os.listdir(random_char)))

            if identity or type(flip_axis) == str:
                angle_dirs = ["Angle_0"]
            else: 
                angle_dirs = sorted(os.listdir(transformed_image_dir))

            logger.info(f"angle dirs: {angle_dirs}")
            angle_pbar = tqdm(angle_dirs, desc="Angles", leave=False)
            for angle_dir in angle_pbar:
                if type(flip_axis) == str:
                    flip_file = os.listdir(transformed_image_dir)[0]
                    flipped_image = os.path.join(transformed_image_dir, flip_file)
                    logger.info(f"Flip file: {flipped_image}")
                    images = [original_image, flipped_image]
                elif identity:
                    images = [original_image, original_image]
                else:
                    angle_file = os.listdir(os.path.join(transformed_image_dir, angle_dir))[0]
                    transformed_image = os.path.join(transformed_image_dir, angle_dir, angle_file)
                    images = [original_image, transformed_image]
                wrong_images = [original_image, random_image]
                for images in [images, wrong_images]:
                    response = model.infer(
                        {
                            "image_paths": images,
                            "text_prompt": text_prompt,
                            "max_pixels": 360 * 420,  # FIXME: What should this be?
                        }
                    )
                    responses["script"].append(script)
                    responses["character_id"].append(char_id)
                    responses["image_id"].append(img_id)
                    responses["angle"].append(int(angle_dir.replace("Angle_", "")))
                    responses["response"].append(response)
                    if type(flip_axis) == str:
                        responses['ground_truth'].append("yes" if images[1] == flipped_image else "no")
                    elif identity:
                         responses['ground_truth'].append("yes" if images[0] == images[1] else "no")
                    else:
                        responses['ground_truth'].append("yes" if images[1] == transformed_image else "no")
                    
                    # Calculate and update metrics in progress bars
                    correct, total, accuracy, recall, tp, fp, tn, fn = calculate_metrics(responses)
                    char_pbar.set_postfix({"Recall": f"{recall:.1f}%", "Acc": f"{accuracy:.1f}%", "TP": tp, "FN": fn})
                    img_pbar.set_postfix({"Recall": f"{recall:.1f}%", "Acc": f"{accuracy:.1f}%", "TP": tp, "FN": fn})
                    angle_pbar.set_postfix({"Recall": f"{recall:.1f}%", "Acc": f"{accuracy:.1f}%", "TP": tp, "FN": fn})
                    
                if "gemini" in model_name.lower():
                    request_count += 1
                    if request_count == 50:
                        logger.info(f"responses saved:\n{responses}")
                        pd.DataFrame(responses).to_csv(os.path.join(output_dir, f"{model_name}_{prompt_name}_{script}.csv"), index=False)
                        exit()
                    time.sleep(15) # to avoid 5 RPM rate limit
            
        logger.info(f"{script} {os.path.basename(char_path)} finished after {(time.time()-start_time)/60} mins")
    pd.DataFrame(responses).to_csv(os.path.join(output_dir, f"{model_name}_{prompt_name}_{script}.csv"), index=False)

    logger.info(f"{script} finished completely after {(time.time()-start_time)/60} minutes")
        
if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    prompt_name = "prompt_3"
    text_prompt = "If I rotate the first image, can I get the second image? Answer in curly brackets, e.g. {Yes} or {No}."
    #text_prompt = "Are these two images the same? Answer in curly brackets, e.g. {Yes} or {No}."
    
    parser = argparse.ArgumentParser(description="Analyze rotation recognition using a vision-language model.")
    parser.add_argument("--script", type=str, help="The script to analyze.")
    parser.add_argument("--prompt_name", type=str, help="The prompt name.")
    parser.add_argument("--text_prompt", type=str, help="The text prompt.")
    parser.add_argument("--model", type=str, help="The model name.")
    parser.add_argument("--flip_axis", type=str, default=None, help="Whether to use boxed images.")
    parser.add_argument("--identity", action='store_true', default=False, help="Whether to perform identity transformation.")
    parser.add_argument("--boxed", action='store_true', default=False, help="Whether to use boxed images.")
    parser.add_argument("--gemini_start_idx", type=int, default=0, help="The starting index for Gemini model.")
    parser.add_argument("--log_path", type=str, default=None, help="The path for the log file.")
    parser.add_argument("--output_dir", type=str, default=None, help="The output directory for responses.")
    parser.add_argument("--data_dir", type=str, default=None, help="The data directory for images.")
    args = parser.parse_args()

    if args.model == "gemini":
        config = {"model_name": "gemini-2.5-pro"}
    else:
        config = {"model_name": "qwen2.5-vl"}

    rotation_recog(args.script, args.prompt_name, args.text_prompt, model_config=config, gemini_start_idx=args.gemini_start_idx, identity=args.identity, boxed=args.boxed, flip_axis=args.flip_axis, log_path=args.log_path, output_dir=args.output_dir, data_dir=args.data_dir)
