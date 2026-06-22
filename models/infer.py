# from transformers import (
#     Qwen2_5_VLForConditionalGeneration,
#     Qwen3VLMoeForConditionalGeneration,
#     Qwen3VLForConditionalGeneration,
#     AutoProcessor,
#     AutoModel,
#     AutoTokenizer,
#     AutoModelForImageTextToText,
#     LlavaForConditionalGeneration,
# )
# from transformers import (
#     Qwen2_5_VLForConditionalGeneration,
#     Qwen3VLMoeForConditionalGeneration,
#     Qwen3VLForConditionalGeneration,
#     AutoProcessor,
#     AutoModel,
#     AutoTokenizer,
#     AutoModelForImageTextToText,
#     LlavaForConditionalGeneration,
# )
import transformers


def maybe_import_transformers(name):
    try:
        return getattr(transformers, name)
    except AttributeError:
        print(f"Skipping unavailable transformers class: {name}")
        return None
    except Exception as e:
        print(f"Error importing transformers.{name}: {e}")
        return None


Qwen2_5_VLForConditionalGeneration = maybe_import_transformers(
    "Qwen2_5_VLForConditionalGeneration"
)
Qwen3VLMoeForConditionalGeneration = maybe_import_transformers(
    "Qwen3VLMoeForConditionalGeneration"
)
Qwen3VLForConditionalGeneration = maybe_import_transformers(
    "Qwen3VLForConditionalGeneration"
)
AutoProcessor = maybe_import_transformers("AutoProcessor")
AutoModel = maybe_import_transformers("AutoModel")
AutoTokenizer = maybe_import_transformers("AutoTokenizer")
AutoModelForImageTextToText = maybe_import_transformers("AutoModelForImageTextToText")
LlavaForConditionalGeneration = maybe_import_transformers(
    "LlavaForConditionalGeneration"
)
from qwen_vl_utils import process_vision_info
from google import genai
from google.genai.types import GenerateContentConfig
import time
from dotenv import load_dotenv
import os
import base64
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import io

from peft import PeftModel
from openai import OpenAI
import re
from typing import Any, Optional, Tuple

load_dotenv()


def _torch_primary_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _hf_model_input_device(model: torch.nn.Module) -> torch.device:
    """Device for batch tensors: match embeddings, never ``meta`` (unreliable with ``device_map``)."""
    try:
        emb = model.get_input_embeddings()
        if emb is not None:
            d = emb.weight.device
            if d.type != "meta":
                return d
    except Exception:
        pass
    return _torch_primary_device()


def _hf_single_gpu_device_map():
    """``device_map`` for loading the full model onto one visible GPU (index 0)."""
    if torch.cuda.is_available():
        return {"": 0}
    return None


class InferenceModel:
    def __init__(self, model_name):
        if model_name == "qwen2.5-vl":
            self.model = QwenInference("Qwen/Qwen2.5-VL-32B-Instruct")
        elif model_name in (
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen/Qwen2.5-VL-32B-Instruct",
            "Qwen/Qwen2.5-VL-72B-Instruct",
        ):
            self.model = QwenInference(model_name)
        elif model_name == "qwen3-vl":
            self.model = Qwen3Inference("Qwen/Qwen3-VL-30B-A3B-Instruct")
        elif model_name.startswith("Qwen/Qwen3-VL-"):
            self.model = Qwen3Inference(model_name)
        elif model_name == "gemini-2.5-pro" or model_name == "gemini-3.1-pro-preview":
            self.model = GeminiInference(model_name)
        elif model_name == "gpt-5.2":
            self.model = GPT52Inference("gpt-5.2")
        elif model_name in ("llava", "llava-1.5-7b"):
            self.model = LlavaInference("llava-hf/llava-1.5-7b-hf")
        elif model_name == "llava-1.5-13b":
            self.model = LlavaInference("llava-hf/llava-1.5-13b-hf")
        elif model_name.startswith("llava-hf/"):
            self.model = LlavaInference(model_name)
        elif model_name == "allenai/Molmo2-8B" or model_name.startswith(
            "allenai/Molmo2"
        ):
            self.model = Molmo2Inference(model_name)
        elif model_name == "OpenGVLab/InternVideo2_5_Chat_8B" or model_name.startswith(
            "OpenGVLab/InternVideo2"
        ):
            self.model = InternVideo2_5_ChatInference(model_name)
        elif model_name == "OpenGVLab/InternVL2_5-8B" or model_name.startswith(
            "OpenGVLab/InternVL2"
        ):
            self.model = InternVL2_5_ChatInference(model_name)
        elif model_name == "OpenGVLab/InternVL3_5-8B" or model_name.startswith(
            "OpenGVLab/InternVL3"
        ):
            self.model = InternVL3_5_ChatInference(model_name)
        elif model_name == "OpenGVLab/InternVL3_5-30B-A3B-HF":
            self.model = InternVL3_5_ChatInference(model_name)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def infer(self, input_data):
        output = self.model.infer(input_data)
        return output


class GeminiInference:
    def __init__(self, model_name):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def _load_image_b64(self, img):
        if isinstance(img, str):
            with open(img, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        elif isinstance(img, Image.Image):
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

    def _build_request(self, image_inputs, text_prompt):
        if not isinstance(image_inputs, (list, tuple)):
            image_inputs = [image_inputs]

        parts = [{"text": text_prompt}]
        for img in image_inputs:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": self._load_image_b64(img),
                    }
                }
            )

        return {
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ]
        }

    def _submit_batch_job(self, batch_requests):
        batch_job = self.client.batches.create(
            model=self.model_name,
            src=batch_requests,
        )
        return batch_job

    def _extract_text(self, resp):
        if resp is None:
            return None
        try:
            return resp.candidates[0].content.parts[0].text
        except Exception:
            return None

    def _poll_batch_job(self, batch_job):
        count = 1
        while True:
            job = self.client.batches.get(name=batch_job.name)
            if "SUCCEEDED" in job.state:
                break
            elif "FAILED" in job.state:
                raise RuntimeError(job.error)
            time.sleep(30)
            count += 1

        return [self._extract_text(r.response) for r in job.dest.inlined_responses]

    def infer_batch(self, batch_inputs):
        """
        Run Gemini batch inference.
        `batch_inputs` is a list of dicts with the same schema as for `infer`,
        but currently only image-based inputs are supported (image_path or image_paths).
        """
        batch_requests = []
        for input_data in batch_inputs:
            if "image_paths" in input_data:
                images = input_data["image_paths"]
            elif "image_path" in input_data:
                images = [input_data["image_path"]]
            else:
                raise ValueError(
                    "Batch inference currently supports inputs with 'image_path' or 'image_paths'."
                )
            text_prompt = input_data["text_prompt"]
            batch_requests.append(self._build_request(images, text_prompt))

        batch_job = self._submit_batch_job(batch_requests)
        return self._poll_batch_job(batch_job)

    def test_batch(self, batch_id):
        job = self.client.batches.get(name=batch_id)
        return [self._extract_text(r.response) for r in job.dest.inlined_responses]

    def upload_video(self, video_file_name):
        video_file = self.client.files.upload(file=video_file_name)

        while video_file.state == "PROCESSING":
            print("Waiting for video to be processed.")
            time.sleep(10)
            video_file = self.client.files.get(name=video_file.name)

        if video_file.state == "FAILED":
            raise ValueError(video_file.state)
        print("Video processing complete: " + video_file.uri)

        return video_file

    def upload_image(self, image_file_name):
        if isinstance(image_file_name, Image.Image):
            # Convert PIL Image to bytes
            buffer = io.BytesIO()
            image_file_name.save(buffer, format="PNG")
            buffer.seek(0)
            image_file = self.client.files.upload(
                file=buffer,
                config={
                    "display_name": "image.png",
                    "mime_type": "image/png",
                },
            )
        elif isinstance(image_file_name, str):
            image_file = self.client.files.upload(file=image_file_name)

        while image_file.state == "PROCESSING":
            print("Waiting for image to be processed.")
            time.sleep(10)
            image_file = self.client.files.get(name=image_file.name)

        if image_file.state == "FAILED":
            raise ValueError(image_file.state)
        print("Image processing complete: " + image_file.uri)

        return image_file

    def prepare_input_image(self, input_data):
        image_file = self.upload_image(input_data["image_path"])
        input_data["image_uploaded"] = image_file
        return input_data

    def prepare_input_video(self, input_data):
        video_file = self.upload_video(input_data["video_path"])
        input_data["video_uploaded"] = video_file
        return input_data

    def _generate_from_images_inline(self, image_inputs, text_prompt: str) -> str:
        """
        Generate from image(s) using inline base64 parts.

        Avoids Files API upload, which can fail SDK validation (fps=[] on images).
        """
        request = self._build_request(image_inputs, text_prompt)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=request["contents"],
            config=GenerateContentConfig(temperature=0),
        )
        return response.text

    def infer(self, input_data):
        # If a list/tuple of inputs is provided, run batch inference using the
        # Gemini Batch API with inlined requests to reduce rate-limit issues.
        if isinstance(input_data, (list, tuple)):
            return self.infer_batch(list(input_data))

        if "image_paths" in input_data:
            return self._generate_from_images_inline(
                input_data["image_paths"], input_data["text_prompt"]
            )
        if "image_path" in input_data:
            return self._generate_from_images_inline(
                input_data["image_path"], input_data["text_prompt"]
            )
        if "video_path" in input_data:
            prepared_inputs = self.prepare_input_video(input_data)
            video = prepared_inputs["video_uploaded"]
            prompt = prepared_inputs["text_prompt"]
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[video, prompt],
                config=GenerateContentConfig(temperature=0),
            )
            return response.text

        raise ValueError(
            "Gemini infer expects 'image_path', 'image_paths', or 'video_path' in input_data."
        )


class GPT52Inference:
    """Inference for gpt-5.2 via OpenAI Responses API with image input."""

    def __init__(self, model_name):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model_name = model_name

    def _image_to_base64_data_uri(self, image_input):
        """Load image from path or PIL Image and return data URI string."""
        if isinstance(image_input, Image.Image):
            buffer = io.BytesIO()
            image_input.save(buffer, format="PNG")
            buffer.seek(0)
            b64 = base64.b64encode(buffer.read()).decode("utf-8")
            return f"data:image/png;base64,{b64}"
        with open(image_input, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def infer(self, input_data):
        prompt = input_data["text_prompt"]
        content = [{"type": "input_text", "text": prompt}]

        if "image_path" in input_data:
            data_uri = self._image_to_base64_data_uri(input_data["image_path"])
            content.append({"type": "input_image", "image_url": data_uri})
        elif "image_paths" in input_data:
            for image_path in input_data["image_paths"]:
                data_uri = self._image_to_base64_data_uri(image_path)
                content.append({"type": "input_image", "image_url": data_uri})
        else:
            raise ValueError("input_data must contain image_path or image_paths")

        response = self.client.responses.create(
            model=self.model_name,
            input=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
        )
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text
        if hasattr(response, "output") and response.output:
            for item in response.output:
                if getattr(item, "content", None):
                    for part in item.content:
                        if getattr(part, "text", None):
                            return part.text
        return str(response)


class LlavaInference:
    """Inference for LLaVA via Hugging Face Transformers."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def infer(self, input_data):
        prompt_text = input_data["text_prompt"]

        if "image_path" in input_data:
            image = input_data["image_path"]
        elif "image_paths" in input_data:
            image_paths = input_data["image_paths"]
            if not image_paths:
                raise ValueError("image_paths is empty")
            if len(image_paths) != 1:
                raise ValueError(
                    "LLaVA does not support multi-image inputs. "
                    "Pass a single image_path (or stitch the images into one)."
                )
            image = image_paths[0]
        else:
            raise ValueError("input_data must contain image_path or image_paths")

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image"},
                ],
            }
        ]

        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.model.device, torch.float16)

        generate_ids = self.model.generate(**inputs, max_new_tokens=128)
        raw_output = self.processor.decode(
            generate_ids[0],
            skip_special_tokens=True,
        ).strip()

        # Post-process to force a pure YES/NO output for downstream experiments.
        upper = raw_output.upper()
        assistant_upper = upper
        marker = "ASSISTANT:"
        if marker in assistant_upper:
            idx = assistant_upper.rfind(marker)
            assistant_upper = assistant_upper[idx + len(marker) :]

        yes_found = re.search(r"\bYES\b", assistant_upper) is not None
        no_found = re.search(r"\bNO\b", assistant_upper) is not None

        if yes_found and not no_found:
            return "YES"
        if no_found and not yes_found:
            return "NO"

        raise ValueError(f"LLaVA output is not a clean YES/NO answer: {raw_output!r}")


def _qwen_processor_video_kwargs(video_kwargs: dict) -> dict:
    """Drop empty fps=[] from image-only inputs (breaks processor validation)."""
    if not video_kwargs:
        return {}
    fps = video_kwargs.get("fps")
    if isinstance(fps, list):
        if len(fps) == 0:
            return {}
        if len(fps) == 1:
            return {"fps": fps[0]}
    return video_kwargs


class QwenInference:
    def __init__(self, model_id):
        load_kwargs = {"torch_dtype": "auto"}
        device_map = _hf_single_gpu_device_map()
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, **load_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=4 * 28 * 28, max_pixels=256 * 28 * 28
        )

    def _to_model_device(self, inputs):
        return inputs.to(_hf_model_input_device(self.model))

    def prepare_input_video(self, input_data):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"{input_data['video_path']}",
                        "max_pixels": input_data["max_pixels"],
                        "fps": input_data["fps"],
                    },
                    {"type": "text", "text": f"{input_data['text_prompt']}"},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, fps=input_data["fps"]
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **_qwen_processor_video_kwargs(video_kwargs),
        )
        return self._to_model_device(inputs)

    def prepare_input_image(self, input_data):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"{input_data['image_path']}",
                        "max_pixels": input_data["max_pixels"],
                    },
                    {"type": "text", "text": f"{input_data['text_prompt']}"},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
            **_qwen_processor_video_kwargs(video_kwargs),
        )
        return self._to_model_device(inputs)

    def prepare_input_images(self, input_data):
        content = []
        max_pixels = input_data.get("max_pixels", None)
        for img_path in input_data["image_paths"]:
            item = {"type": "image", "image": img_path}
            if max_pixels is not None:
                item["max_pixels"] = max_pixels
            content.append(item)

        content.append({"type": "text", "text": f"{input_data['text_prompt']}"})

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
            **_qwen_processor_video_kwargs(video_kwargs),
        )
        return self._to_model_device(inputs)

    def _prepare_inputs(self, input_data):
        if "image_path" in input_data:
            return self.prepare_input_image(input_data)
        if "video_path" in input_data:
            return self.prepare_input_video(input_data)
        if "image_paths" in input_data:
            return self.prepare_input_images(input_data)
        raise ValueError(
            "input_data must contain image_path, image_paths, or video_path"
        )

    @staticmethod
    def _yes_no_token_ids(tokenizer) -> Tuple[list, list]:
        """Token ids whose decoded form is a standalone Yes/No answer variant."""
        candidates_yes = ("Yes", "yes", "YES", "{Yes}", "{yes}", " Yes", " yes")
        candidates_no = ("No", "no", "NO", "{No}", "{no}", " No", " no")
        yes_ids, no_ids = [], []
        for text in candidates_yes:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in yes_ids:
                yes_ids.append(ids[0])
        for text in candidates_no:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in no_ids:
                no_ids.append(ids[0])
        return yes_ids, no_ids

    def infer_yes_no_probs(self, input_data) -> dict:
        """
        First-token logits for Yes/No after the prompt (calibration analysis).

        Returns dict with p_yes, p_no (sums over token variants), p_yes_normalized,
        yes_token_probs, no_token_probs, and yes_ids/no_ids used.
        """
        prepared_inputs = self._prepare_inputs(input_data)
        tokenizer = self.processor.tokenizer
        yes_ids, no_ids = self._yes_no_token_ids(tokenizer)

        with torch.no_grad():
            outputs = self.model(**prepared_inputs)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)[0]

        p_yes = sum(probs[tid].item() for tid in yes_ids)
        p_no = sum(probs[tid].item() for tid in no_ids)
        denom = p_yes + p_no
        p_yes_normalized = (p_yes / denom) if denom > 0 else float("nan")

        return {
            "p_yes": p_yes,
            "p_no": p_no,
            "p_yes_normalized": p_yes_normalized,
            "yes_token_probs": {int(tid): probs[tid].item() for tid in yes_ids},
            "no_token_probs": {int(tid): probs[tid].item() for tid in no_ids},
            "yes_token_ids": yes_ids,
            "no_token_ids": no_ids,
        }

    def infer(self, input_data):
        prepared_inputs = self._prepare_inputs(input_data)
        generated_ids = self.model.generate(**prepared_inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(prepared_inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text

    def set_finetune_mode(self, finetune_dir):
        self.model = PeftModel.from_pretrained(self.model, finetune_dir)
        self.model.eval()


class Qwen3Inference:
    """Inference for Qwen3-VL (e.g. Qwen3-VL-30B-A3B-Instruct) using apply_chat_template API."""

    def __init__(self, model_id):
        if "Qwen3-VL-8B" in model_id or model_id.endswith("Qwen3-VL-8B-Instruct"):
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id, dtype="auto", device_map="auto"
            )
        else:
            self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_id, torch_dtype="auto", device_map="auto"
            )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def _to_device(self, inputs):
        """Move input tensors to the model device."""
        if hasattr(inputs, "to"):
            return inputs.to(self.model.device)
        return {
            k: v.to(self.model.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

    def prepare_input_image(self, input_data):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": input_data["image_path"]},
                    {"type": "text", "text": input_data["text_prompt"]},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return self._to_device(inputs)

    def prepare_input_images(self, input_data):
        content = []
        for img_path in input_data["image_paths"]:
            content.append({"type": "image", "image": img_path})
        content.append({"type": "text", "text": input_data["text_prompt"]})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return self._to_device(inputs)

    def prepare_input_video(self, input_data):
        raise NotImplementedError(
            "Qwen3Inference: video input not implemented; use image_path or image_paths."
        )

    def _prepare_inputs(self, input_data):
        if "image_path" in input_data:
            return self.prepare_input_image(input_data)
        if "image_paths" in input_data:
            return self.prepare_input_images(input_data)
        if "video_path" in input_data:
            return self.prepare_input_video(input_data)
        raise ValueError(
            "input_data must contain image_path, image_paths, or video_path"
        )

    def infer_yes_no_probs(self, input_data) -> dict:
        """First-token Yes/No logits (same contract as QwenInference)."""
        prepared_inputs = self._prepare_inputs(input_data)
        tokenizer = self.processor.tokenizer
        yes_ids, no_ids = QwenInference._yes_no_token_ids(tokenizer)

        with torch.no_grad():
            outputs = self.model(**prepared_inputs)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)[0]

        p_yes = sum(probs[tid].item() for tid in yes_ids)
        p_no = sum(probs[tid].item() for tid in no_ids)
        denom = p_yes + p_no
        p_yes_normalized = (p_yes / denom) if denom > 0 else float("nan")

        return {
            "p_yes": p_yes,
            "p_no": p_no,
            "p_yes_normalized": p_yes_normalized,
            "yes_token_probs": {int(tid): probs[tid].item() for tid in yes_ids},
            "no_token_probs": {int(tid): probs[tid].item() for tid in no_ids},
            "yes_token_ids": yes_ids,
            "no_token_ids": no_ids,
        }

    def infer(self, input_data):
        if "image_path" in input_data:
            prepared_inputs = self.prepare_input_image(input_data)
        elif "video_path" in input_data:
            prepared_inputs = self.prepare_input_video(input_data)
        elif "image_paths" in input_data:
            prepared_inputs = self.prepare_input_images(input_data)
        else:
            raise ValueError(
                "input_data must contain image_path, image_paths, or video_path"
            )

        input_ids = (
            prepared_inputs["input_ids"]
            if isinstance(prepared_inputs, dict)
            else prepared_inputs.input_ids
        )
        generated_ids = self.model.generate(**prepared_inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text

    def set_finetune_mode(self, finetune_dir):
        self.model = PeftModel.from_pretrained(self.model, finetune_dir)
        self.model.eval()


class InternVideo2_5_ChatInference:
    """InternVideo2.5-Chat (OpenGVLab) via ``AutoModel.chat`` — preprocessing from the official recipe."""

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, model_id: str):
        try:
            from decord import VideoReader, cpu as decord_cpu
        except ImportError as exc:
            raise RuntimeError(
                "InternVideo inference requires decord. Install with: pip install decord"
            ) from exc
        self._VideoReader = VideoReader
        self._decord_cpu = decord_cpu
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "InternVideo2.5-Chat inference requires CUDA (model is loaded on GPU)."
            )
        self.model = self.model.cuda()
        self.model.eval()

    @staticmethod
    def build_transform(input_size: int) -> T.Compose:
        mean, std = (
            InternVideo2_5_ChatInference.IMAGENET_MEAN,
            InternVideo2_5_ChatInference.IMAGENET_STD,
        )
        return T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (input_size, input_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    @staticmethod
    def find_closest_aspect_ratio(
        aspect_ratio: float,
        target_ratios: list,
        width: int,
        height: int,
        image_size: int,
    ) -> tuple:
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    @classmethod
    def dynamic_preprocess(
        cls,
        image: Image.Image,
        min_num: int = 1,
        max_num: int = 6,
        image_size: int = 448,
        use_thumbnail: bool = False,
    ) -> list:
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        }
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        target_aspect_ratio = cls.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        tw = target_width // image_size
        for i in range(blocks):
            col = i % tw
            row = i // tw
            box = (
                col * image_size,
                row * image_size,
                (col + 1) * image_size,
                (row + 1) * image_size,
            )
            processed_images.append(resized_img.crop(box))
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image: Image.Image, input_size: int = 448, max_num: int = 6):
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(
            image, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(im) for im in images]
        return torch.stack(pixel_values)

    @staticmethod
    def get_index(
        bound: Optional[Tuple[float, float]],
        fps: float,
        max_frame: int,
        first_idx: int = 0,
        num_segments: int = 32,
    ):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array(
            [
                int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
                for idx in range(num_segments)
            ]
        )
        return frame_indices

    @staticmethod
    def get_num_frames_by_duration(duration: float) -> int:
        local_num_frames = 4
        num_segments = int(duration // local_num_frames)
        if num_segments == 0:
            num_frames = local_num_frames
        else:
            num_frames = local_num_frames * num_segments
        num_frames = min(512, num_frames)
        num_frames = max(128, num_frames)
        return num_frames

    def load_video(
        self,
        video_path: str,
        bound: Optional[Tuple[float, float]] = None,
        input_size: int = 448,
        max_num: int = 1,
        num_segments: int = 32,
        get_frame_by_duration: bool = False,
    ):
        vr = self._VideoReader(video_path, ctx=self._decord_cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        pixel_values_list: list = []
        num_patches_list: list = []
        transform = self.build_transform(input_size=input_size)
        if get_frame_by_duration:
            duration = max_frame / fps
            num_segments = self.get_num_frames_by_duration(duration)
        frame_indices = self.get_index(
            bound, fps, max_frame, first_idx=0, num_segments=num_segments
        )
        for frame_index in frame_indices:
            idx = int(np.clip(frame_index, 0, max_frame))
            img = Image.fromarray(vr[idx].asnumpy()).convert("RGB")
            tiles = self.dynamic_preprocess(
                img, image_size=input_size, use_thumbnail=True, max_num=max_num
            )
            pixel_values = torch.stack([transform(t) for t in tiles])
            num_patches_list.append(int(pixel_values.shape[0]))
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

    def _default_generation_config(self, max_new_tokens: int) -> dict:
        return {
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": int(max_new_tokens),
            "top_p": 0.1,
            "num_beams": 1,
        }

    def infer(self, input_data: dict) -> list:
        max_new_tokens = int(input_data.get("max_new_tokens", 1024))
        gen_cfg = dict(self._default_generation_config(max_new_tokens))
        if isinstance(input_data.get("generation_config"), dict):
            gen_cfg.update(input_data["generation_config"])

        input_size = int(input_data.get("intern_input_size", 448))
        max_num = int(input_data.get("intern_max_num", 1))
        num_segments = int(input_data.get("num_segments", 128))
        get_frame_by_duration = bool(
            input_data.get("intern_get_frame_by_duration", False)
        )
        bound = input_data.get("bound")
        if bound is not None and (
            not isinstance(bound, (list, tuple)) or len(bound) != 2
        ):
            bound = None

        device = next(self.model.parameters()).device

        if "video_path" in input_data:
            pixel_values, num_patches_list = self.load_video(
                str(input_data["video_path"]),
                bound=tuple(bound) if bound is not None else None,
                input_size=input_size,
                max_num=max_num,
                num_segments=num_segments,
                get_frame_by_duration=get_frame_by_duration,
            )
        elif "image_path" in input_data:
            img = Image.open(input_data["image_path"]).convert("RGB")
            pixel_values = self.load_image(img, input_size=input_size, max_num=max_num)
            num_patches_list = [int(pixel_values.shape[0])]
        elif "image_paths" in input_data:
            paths = input_data["image_paths"]
            if not paths:
                raise ValueError("image_paths is empty")
            img = Image.open(paths[0]).convert("RGB")
            pixel_values = self.load_image(img, input_size=input_size, max_num=max_num)
            num_patches_list = [int(pixel_values.shape[0])]
        else:
            raise ValueError(
                "InternVideo2.5-Chat infer requires video_path, image_path, or image_paths"
            )

        pixel_values = pixel_values.to(dtype=torch.bfloat16, device=device)
        video_prefix = "".join(
            [f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))]
        )
        question = video_prefix + str(input_data.get("text_prompt", ""))

        with torch.inference_mode():
            output_text, _ = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                gen_cfg,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True,
            )
        return (
            [output_text.strip()]
            if isinstance(output_text, str)
            else [str(output_text)]
        )


class InternVL2_5_ChatInference(InternVideo2_5_ChatInference):
    """InternVL2.5 (OpenGVLab) via ``AutoModel.chat`` — official image/video preprocessing."""

    def __init__(self, model_id: str):
        try:
            from decord import VideoReader, cpu as decord_cpu
        except ImportError as exc:
            raise RuntimeError(
                "InternVL inference requires decord. Install with: pip install decord"
            ) from exc
        self._VideoReader = VideoReader
        self._decord_cpu = decord_cpu
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
        )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "InternVL2.5 inference requires CUDA (model is loaded on GPU)."
            )
        self.model = self.model.cuda()
        self.model.eval()

    @staticmethod
    def _build_question(
        text_prompt: str,
        num_images: int,
        *,
        video_frames: bool = False,
    ) -> str:
        if video_frames:
            prefix = "".join(f"Frame{i + 1}: <image>\n" for i in range(num_images))
        elif num_images == 1:
            prefix = "<image>\n"
        else:
            prefix = "".join(f"Image-{i + 1}: <image>\n" for i in range(num_images))
        return prefix + str(text_prompt)

    def infer(self, input_data: dict) -> list:
        max_new_tokens = int(input_data.get("max_new_tokens", 1024))
        gen_cfg = dict(self._default_generation_config(max_new_tokens))
        if isinstance(input_data.get("generation_config"), dict):
            gen_cfg.update(input_data["generation_config"])

        input_size = int(input_data.get("intern_input_size", 448))
        text_prompt = str(input_data.get("text_prompt", ""))
        bound = input_data.get("bound")
        if bound is not None and (
            not isinstance(bound, (list, tuple)) or len(bound) != 2
        ):
            bound = None

        device = next(self.model.parameters()).device

        if "video_path" in input_data:
            max_num = int(input_data.get("intern_max_num", 1))
            num_segments = int(input_data.get("num_segments", 8))
            get_frame_by_duration = bool(
                input_data.get("intern_get_frame_by_duration", False)
            )
            pixel_values, num_patches_list = self.load_video(
                str(input_data["video_path"]),
                bound=tuple(bound) if bound is not None else None,
                input_size=input_size,
                max_num=max_num,
                num_segments=num_segments,
                get_frame_by_duration=get_frame_by_duration,
            )
            question = self._build_question(
                text_prompt, len(num_patches_list), video_frames=True
            )
        elif "image_path" in input_data:
            max_num = int(input_data.get("intern_max_num", 12))
            img = Image.open(input_data["image_path"]).convert("RGB")
            pixel_values = self.load_image(img, input_size=input_size, max_num=max_num)
            num_patches_list = [int(pixel_values.shape[0])]
            question = self._build_question(text_prompt, 1)
        elif "image_paths" in input_data:
            max_num = int(input_data.get("intern_max_num", 12))
            paths = input_data["image_paths"]
            if not paths:
                raise ValueError("image_paths is empty")
            pixel_values_list = []
            num_patches_list = []
            for path in paths:
                img = Image.open(path).convert("RGB")
                pv = self.load_image(img, input_size=input_size, max_num=max_num)
                pixel_values_list.append(pv)
                num_patches_list.append(int(pv.shape[0]))
            pixel_values = torch.cat(pixel_values_list)
            question = self._build_question(text_prompt, len(paths))
        else:
            raise ValueError(
                "InternVL2.5 infer requires video_path, image_path, or image_paths"
            )

        pixel_values = pixel_values.to(dtype=torch.bfloat16, device=device)

        with torch.inference_mode():
            output_text, _ = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                gen_cfg,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True,
            )
        return (
            [output_text.strip()]
            if isinstance(output_text, str)
            else [str(output_text)]
        )


class InternVL3_5_ChatInference(InternVL2_5_ChatInference):
    """InternVL3.5 (OpenGVLab) via ``AutoModel.chat`` — official image/video preprocessing."""

    def __init__(self, model_id: str):
        try:
            from decord import VideoReader, cpu as decord_cpu
        except ImportError as exc:
            raise RuntimeError(
                "InternVL inference requires decord. Install with: pip install decord"
            ) from exc
        self._VideoReader = VideoReader
        self._decord_cpu = decord_cpu
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False
        )
        load_kwargs = dict(
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
        )
        device_map = _hf_single_gpu_device_map()
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        self.model = AutoModel.from_pretrained(model_id, **load_kwargs).eval()
        if device_map is None:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "InternVL3.5 inference requires CUDA (model is loaded on GPU)."
                )
            self.model = self.model.cuda()


class Molmo2Inference:
    """Inference for Ai2 Molmo2 (e.g. allenai/Molmo2-8B) via AutoModelForImageTextToText."""

    def __init__(self, model_id):
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        dm = _hf_single_gpu_device_map()
        if dm is not None:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map=dm,
            )
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype="auto",
            )

    def _move_tensors_to_device(self, obj: Any, device: torch.device) -> Any:
        if torch.is_tensor(obj):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: self._move_tensors_to_device(v, device) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._move_tensors_to_device(x, device) for x in obj)
        return obj

    def _inputs_to_device(self, inputs):
        device = _hf_model_input_device(self.model)
        to_fn = getattr(inputs, "to", None)
        if callable(to_fn) and not isinstance(inputs, dict):
            return to_fn(device)
        if isinstance(inputs, dict):
            return self._move_tensors_to_device(inputs, device)
        return self._move_tensors_to_device(inputs, device)

    def _messages_to_inputs(self, messages):
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        return self._inputs_to_device(inputs)

    def _generate(self, inputs, max_new_tokens):
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        input_len = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[0, input_len:]
        text = self.processor.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()
        return [text]

    def prepare_input_video(self, input_data):
        video_item = {"type": "video", "video": str(input_data["video_path"])}
        if "fps" in input_data:
            video_item["fps"] = float(input_data["fps"])
        if "max_pixels" in input_data:
            video_item["max_pixels"] = int(input_data["max_pixels"])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": input_data["text_prompt"]},
                    video_item,
                ],
            }
        ]
        return self._messages_to_inputs(messages)

    def prepare_input_image(self, input_data):
        img_item = {"type": "image", "image": str(input_data["image_path"])}
        if "max_pixels" in input_data:
            img_item["max_pixels"] = int(input_data["max_pixels"])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": input_data["text_prompt"]},
                    img_item,
                ],
            }
        ]
        return self._messages_to_inputs(messages)

    def prepare_input_images(self, input_data):
        content = [{"type": "text", "text": input_data["text_prompt"]}]
        max_pixels = input_data.get("max_pixels")
        for path in input_data["image_paths"]:
            item = {"type": "image", "image": str(path)}
            if max_pixels is not None:
                item["max_pixels"] = int(max_pixels)
            content.append(item)
        messages = [{"role": "user", "content": content}]
        return self._messages_to_inputs(messages)

    def prepare_input_text(self, input_data):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": input_data["text_prompt"]}],
            }
        ]
        return self._messages_to_inputs(messages)

    def infer(self, input_data):
        if "video_path" in input_data:
            inputs = self.prepare_input_video(input_data)
        elif "image_path" in input_data:
            inputs = self.prepare_input_image(input_data)
        elif "image_paths" in input_data:
            inputs = self.prepare_input_images(input_data)
        elif "text_prompt" in input_data:
            inputs = self.prepare_input_text(input_data)
        else:
            raise ValueError(
                "input_data must contain video_path, image_path, image_paths, or text_prompt"
            )
        max_new_tokens = input_data.get("max_new_tokens", 2048)
        return self._generate(inputs, max_new_tokens)
