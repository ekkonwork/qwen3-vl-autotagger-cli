import base64
import io
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image

try:
    from transformers import AutoProcessor, BitsAndBytesConfig
except Exception as e:  # pragma: no cover
    AutoProcessor = None
    BitsAndBytesConfig = None
    _transformers_import_error = e
else:
    _transformers_import_error = None

try:
    from transformers import Qwen3VLForConditionalGeneration
except Exception as e:  # pragma: no cover
    Qwen3VLForConditionalGeneration = None
    _qwen3vl_import_error = e
else:
    _qwen3vl_import_error = None

try:
    from qwen_vl_utils import process_vision_info
except Exception as e:  # pragma: no cover
    process_vision_info = None
    _qwen_import_error = e
else:
    _qwen_import_error = None


DEFAULT_PROMPT = (
    "You are an Adobe Stock metadata generator.\n"
    "Return ONLY a valid JSON object.\n"
    "Structure: {\"title\": \"English title max 150 chars\", \"keywords\": [\"tag1\", \"tag2\", \"tag3\"]}\n"
    "Rules:\n"
    "1. GENERATE ~60 KEYWORDS. Include synonyms, visual details, concepts.\n"
    "2. PREFER SINGLE WORDS where possible.\n"
    "3. Output raw JSON only.\n"
)

DEFAULT_EXTENSIONS = ("jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff")
MODEL_CACHE: Dict[Tuple[str, int, int, bool, str, bool], Tuple[object, object, torch.device]] = {}


def _pil_to_data_url(pil: Image.Image) -> str:
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _extract_and_fix_json(text: str):
    text = text.replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    start = text.find("{")
    if start != -1:
        json_fragment = text[start:]
        if "}" not in json_fragment and '"keywords": [' in json_fragment:
            fixed_json = json_fragment.rstrip(", ") + '"]}'
            fixed_json = fixed_json.replace('""', '"')
            try:
                return json.loads(fixed_json)
            except Exception:
                pass

    return None


def _clean_split_and_limit(keywords_value, limit: int = 50) -> List[str]:
    if isinstance(keywords_value, str):
        keywords_iterable: Sequence[str] = [x.strip() for x in keywords_value.split(",")]
    elif isinstance(keywords_value, Sequence):
        keywords_iterable = [str(x) for x in keywords_value]
    else:
        keywords_iterable = [str(keywords_value)]

    final_list: List[str] = []
    for item in keywords_iterable:
        clean_item = str(item).replace("_", " ").replace("-", " ")
        words = clean_item.split(" ")
        for word in words:
            word = word.strip().lower()
            if len(word) > 2:
                final_list.append(word)

    seen = set()
    unique_list = []
    for word in final_list:
        if word not in seen:
            unique_list.append(word)
            seen.add(word)

    return unique_list[: int(limit)]


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_model_reference(model_id: str, auto_download: bool, local_model_path: str) -> str:
    if auto_download:
        return (model_id or "").strip()
    if local_model_path:
        return os.path.expanduser(local_model_path.strip())
    return (model_id or "").strip()


def load_model_and_processor(
    model_ref: str,
    min_pixels: int,
    max_pixels: int,
    load_in_4bit: bool,
    allow_download: bool,
):
    device = _get_device()
    cache_key = (model_ref, min_pixels, max_pixels, load_in_4bit, device.type, allow_download)
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    if AutoProcessor is None:
        raise RuntimeError(f"transformers is not available in this environment: {_transformers_import_error}")
    if process_vision_info is None:
        raise RuntimeError(f"qwen-vl-utils is not available: {_qwen_import_error}")
    if Qwen3VLForConditionalGeneration is None:
        raise RuntimeError(
            "Qwen3VLForConditionalGeneration is not available in this transformers build. "
            f"Import error: {_qwen3vl_import_error}"
        )

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    device_map = "auto" if device.type == "cuda" else None

    quant_config = None
    if load_in_4bit:
        if device.type != "cuda":
            print("4-bit quantization requires CUDA; loading without quantization.")
        elif BitsAndBytesConfig is None:
            print("bitsandbytes is not available; loading without 4-bit quantization.")
        else:
            try:
                quant_config = BitsAndBytesConfig(load_in_4bit=True)
            except Exception:
                quant_config = None

    model = None
    primary_load_error = None
    local_files_only = not allow_download

    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_ref,
            torch_dtype=dtype,
            device_map=device_map,
            quantization_config=quant_config,
            local_files_only=local_files_only,
        )
    except Exception as e:
        primary_load_error = e

    if model is None and quant_config is not None:
        print("[qwen3-vl-autotagger-cli] 4-bit load failed; retrying without 4-bit quantization.")
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_ref,
                torch_dtype=dtype,
                device_map=device_map,
                quantization_config=None,
                local_files_only=local_files_only,
            )
            print("[qwen3-vl-autotagger-cli] Model loaded without 4-bit quantization.")
        except Exception as retry_error:
            raise RuntimeError(
                "Failed to load Qwen3-VL model.\n"
                f"Primary error: {type(primary_load_error).__name__}: {primary_load_error}\n"
                f"Retry without 4-bit error: {type(retry_error).__name__}: {retry_error}"
            ) from retry_error

    if model is None:
        raise RuntimeError(
            "Failed to load Qwen3-VL model.\n"
            f"Error: {type(primary_load_error).__name__}: {primary_load_error}"
        ) from primary_load_error

    if device_map is None:
        model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        model_ref,
        min_pixels=int(min_pixels),
        max_pixels=int(max_pixels),
        local_files_only=local_files_only,
    )

    MODEL_CACHE[cache_key] = (model, processor, device)
    return model, processor, device


def _normalize_output_format(fmt: str) -> Tuple[str, str]:
    ext = (fmt or "png").strip().lower()
    if ext in {"jpg", "jpeg"}:
        return "jpg", "JPEG"
    if ext == "webp":
        return "webp", "WEBP"
    return "png", "PNG"


def _build_unique_output_path(output_dir: str, prefix: str, index: int, ext: str) -> str:
    safe_prefix = re.sub(r"[^a-zA-Z0-9_-]", "_", prefix) if prefix else "autotag"
    next_index = max(0, int(index))
    while True:
        filename = f"{safe_prefix}_{next_index:05d}.{ext}"
        path = os.path.join(output_dir, filename)
        if not os.path.exists(path):
            return path
        next_index += 1


def save_with_xmp(
    pil: Image.Image,
    title: str,
    keywords: List[str],
    output_dir: str,
    prefix: str,
    index: int,
    fmt: str,
    require_exiftool: bool,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    file_ext, save_format = _normalize_output_format(fmt)
    path = _build_unique_output_path(output_dir, prefix, index, file_ext)

    save_kwargs = {}
    if save_format == "JPEG":
        save_kwargs["quality"] = 95
        save_kwargs["subsampling"] = 0
    elif save_format == "WEBP":
        save_kwargs["quality"] = 95

    pil.save(path, format=save_format, **save_kwargs)

    if shutil.which("exiftool") is None:
        msg = "exiftool not found in PATH; skipping XMP embedding"
        if require_exiftool:
            raise RuntimeError(msg)
        print(msg)
        return path

    safe_title = title[:199] if len(title) > 199 else title
    keywords_str = ", ".join(keywords)

    cmd = [
        "exiftool",
        "-overwrite_original",
        "-codedcharacterset=utf8",
        "-m",
        f"-XMP-dc:Title={safe_title}",
        f"-XMP-dc:Description={safe_title}",
        "-sep",
        ", ",
        f"-XMP-dc:Subject={keywords_str}",
        path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0 and require_exiftool:
        err = result.stderr.decode("utf-8", errors="ignore") if result.stderr else "Unknown error"
        raise RuntimeError(f"exiftool failed: {err}")

    return path


def collect_input_images(input_path: str, recursive: bool, extensions: Sequence[str]) -> List[Path]:
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Input path not found: {source}")

    ext_set = {("." + ext.lower().lstrip(".")) for ext in extensions}
    if source.is_file():
        if source.suffix.lower() in ext_set:
            return [source]
        raise ValueError(f"Input file extension is not supported: {source.suffix}")

    pattern = "**/*" if recursive else "*"
    images = []
    for path in source.glob(pattern):
        if path.is_file() and path.suffix.lower() in ext_set:
            images.append(path)

    return sorted(images)


def generate_tags_for_image(
    pil: Image.Image,
    model,
    processor,
    system_prompt: str,
    max_keywords: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    attempts: int,
    min_pixels: int,
    max_pixels: int,
    allow_resize: bool,
):
    b64 = _pil_to_data_url(pil)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": b64, "max_pixels": int(max_pixels), "min_pixels": int(min_pixels)},
                {"type": "text", "text": system_prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    try:
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            do_resize=bool(allow_resize),
        )
    except TypeError:
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

    if hasattr(model, "device"):
        inputs = inputs.to(model.device)

    last_error = None
    output_text = ""
    for attempt in range(int(attempts)):
        try:
            do_sample = attempt > 0 and temperature > 0
            gen_kwargs = {
                "max_new_tokens": int(max_new_tokens),
                "repetition_penalty": float(repetition_penalty),
                "do_sample": bool(do_sample),
            }
            if do_sample:
                gen_kwargs["temperature"] = float(temperature)
                gen_kwargs["top_p"] = float(top_p)

            with torch.inference_mode():
                generated_ids = model.generate(**inputs, **gen_kwargs)

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            data = _extract_and_fix_json(output_text)
            if data and data.get("title") and data.get("keywords") is not None:
                title = str(data["title"]).strip()
                final_keywords = _clean_split_and_limit(data["keywords"], limit=int(max_keywords))
                raw_json = json.dumps({"title": title, "keywords": final_keywords}, ensure_ascii=False)
                if len(final_keywords) < 5 and attempt < (int(attempts) - 1):
                    continue
                return title, final_keywords, raw_json
        except Exception as e:
            last_error = e
            if attempt == int(attempts) - 1:
                if "out of memory" in str(e).lower() and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            time.sleep(1)

    raise RuntimeError(
        "Failed to generate valid metadata JSON.\n"
        f"Last error: {type(last_error).__name__}: {last_error}\n"
        f"Model output snippet: {output_text[:500]}"
    ) from last_error

