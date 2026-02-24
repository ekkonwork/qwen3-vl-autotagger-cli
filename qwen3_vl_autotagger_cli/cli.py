import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image

from .core import (
    DEFAULT_EXTENSIONS,
    DEFAULT_PROMPT,
    collect_input_images,
    generate_tags_for_image,
    load_model_and_processor,
    resolve_model_reference,
    save_with_xmp,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen3-vl-autotagger",
        description="Generate Adobe Stock-style title + keywords with Qwen3-VL and optionally embed XMP.",
    )

    parser.add_argument("input_path", help="Input file or directory with images.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input directory.")
    parser.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help="Comma-separated image extensions to include (default: jpg,jpeg,png,webp,bmp,tif,tiff).",
    )

    parser.add_argument("--system-prompt", default=DEFAULT_PROMPT, help="System prompt that enforces JSON output.")
    parser.add_argument("--max-keywords", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=756 * 756)
    parser.add_argument(
        "--allow-resize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow processor resize to valid patch sizes and pixel budget.",
    )

    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--auto-download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow model download from Hugging Face.",
    )
    parser.add_argument(
        "--local-model-path",
        default="",
        help="Path to a local model folder (used when --no-auto-download).",
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use 4-bit quantization when available on CUDA.",
    )

    parser.add_argument(
        "--write-xmp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save images and embed XMP via exiftool.",
    )
    parser.add_argument(
        "--require-exiftool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if exiftool is missing when writing XMP.",
    )
    parser.add_argument("--output-dir", default="outputs", help="Directory for saved tagged images.")
    parser.add_argument("--output-format", default="jpg", help="Output format: jpg, png, webp.")
    parser.add_argument("--file-prefix", default="qwen3_autotag", help="Output filename prefix.")
    parser.add_argument(
        "--metadata-jsonl",
        default="",
        help="Metadata JSONL path. Default: <output-dir>/metadata.jsonl",
    )
    parser.add_argument(
        "--overwrite-metadata-jsonl",
        action="store_true",
        help="Overwrite metadata JSONL instead of appending.",
    )
    parser.add_argument("--log-tags", action="store_true", help="Print title and keyword preview per image.")

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    model_ref = resolve_model_reference(args.model_id, args.auto_download, args.local_model_path)
    if not model_ref:
        parser.error("Model is not set. Provide --model-id or --local-model-path.")
    if not args.auto_download and args.local_model_path and not os.path.isdir(args.local_model_path):
        parser.error(f"Local model path not found: {args.local_model_path}")

    extensions = [x.strip() for x in args.extensions.split(",") if x.strip()]
    images = collect_input_images(args.input_path, args.recursive, extensions)
    if not images:
        parser.error("No images found for the given path/extensions.")

    metadata_jsonl = Path(args.metadata_jsonl) if args.metadata_jsonl else Path(args.output_dir) / "metadata.jsonl"
    metadata_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite_metadata_jsonl else "a"

    print(f"[qwen3-vl-autotagger-cli] Loading model: {model_ref}")
    model, processor, device = load_model_and_processor(
        model_ref=model_ref,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        load_in_4bit=args.load_in_4bit,
        allow_download=bool(args.auto_download),
    )
    print(f"[qwen3-vl-autotagger-cli] Device: {device.type}")
    print(f"[qwen3-vl-autotagger-cli] Found images: {len(images)}")

    success = 0
    failed = 0
    with metadata_jsonl.open(mode, encoding="utf-8") as out:
        for idx, image_path in enumerate(images):
            try:
                with Image.open(image_path) as img:
                    pil = img.convert("RGB")

                title, keywords, raw_json = generate_tags_for_image(
                    pil=pil,
                    model=model,
                    processor=processor,
                    system_prompt=args.system_prompt,
                    max_keywords=args.max_keywords,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    attempts=args.attempts,
                    min_pixels=args.min_pixels,
                    max_pixels=args.max_pixels,
                    allow_resize=args.allow_resize,
                )

                saved_path = ""
                if args.write_xmp:
                    saved_path = save_with_xmp(
                        pil=pil,
                        title=title,
                        keywords=keywords,
                        output_dir=args.output_dir,
                        prefix=args.file_prefix,
                        index=idx,
                        fmt=args.output_format,
                        require_exiftool=args.require_exiftool,
                    )

                record = {
                    "input_path": str(image_path),
                    "output_path": saved_path,
                    "title": title,
                    "keywords": keywords,
                    "json": json.loads(raw_json),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()

                if args.log_tags:
                    preview = ", ".join(keywords[:20])
                    print(
                        f"[{idx + 1}/{len(images)}] OK {image_path.name} | "
                        f"title='{title}' | keywords({len(keywords)}): {preview}"
                    )
                else:
                    print(f"[{idx + 1}/{len(images)}] OK {image_path.name}")
                success += 1
            except Exception as e:
                failed += 1
                print(f"[{idx + 1}/{len(images)}] FAIL {image_path.name}: {e}")

    print(
        f"[qwen3-vl-autotagger-cli] Done. success={success}, failed={failed}, "
        f"metadata_jsonl={metadata_jsonl}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

