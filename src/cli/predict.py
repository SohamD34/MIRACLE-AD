"""Run recording-level inference and optionally expose MIL attention weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..data import load_audio_bag
from ..models import available_backbones, available_networks, get_model
from ..split_registry import Language


def _load_config(checkpoint: Path, explicit: str | None) -> dict:
    path = Path(explicit) if explicit else checkpoint.parent / "run_config.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _pick(args, config, name, *, required=False, default=None):
    value = getattr(args, name)
    if value is None:
        value = config.get(name, default)
    if required and value is None:
        raise ValueError(f"Missing --{name.replace('_', '-')} and no value was found in run_config.json")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run MIRACLE-AD inference on one audio file")
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--backbone", choices=available_backbones(), default=None)
    parser.add_argument("--network", choices=available_networks(), default=None)
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--chunk-duration", type=float, default=None)
    parser.add_argument("--overlap-factor", type=float, default=None)
    parser.add_argument("--combine-mci-ad", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--language-aware", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args(argv)

    checkpoint_path = Path(args.checkpoint)
    config = _load_config(checkpoint_path, args.config)
    try:
        backbone = _pick(args, config, "backbone", required=True)
        network = _pick(args, config, "network", required=True)
        sample_rate = int(_pick(args, config, "sample_rate", default=24000))
        chunk_duration = float(_pick(args, config, "chunk_duration", default=10.0))
        overlap_factor = float(_pick(args, config, "overlap_factor", default=0.5))
        combine_mci_ad = bool(_pick(args, config, "combine_mci_ad", default=False))
        language_aware = bool(_pick(args, config, "language_aware", default=False))
    except ValueError as exc:
        parser.error(str(exc))

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    device = torch.device(device_name)
    model = get_model(
        backbone_type=backbone,
        network_type=network,
        num_disease_classes=2 if combine_mci_ad else 3,
        num_language_classes=len(Language),
        sample_rate=sample_rate,
        lang_aware=language_aware,
        wav2vec2_model=config.get("wav2vec2_model", "facebook/wav2vec2-base-960h"),
        cache_dir=config.get("cache_dir"),
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    chunks, ranges = load_audio_bag(
        args.input,
        sample_rate=sample_rate,
        chunk_duration=chunk_duration,
        overlap_factor=overlap_factor,
    )
    inputs = torch.from_numpy(chunks).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(inputs, return_attention=True)

    attention = None
    language_logits = None
    if language_aware:
        heads, attention = output
        disease_logits, language_logits = heads
    elif isinstance(output, tuple):
        disease_logits, attention = output
    else:
        disease_logits = output

    disease_probabilities = torch.softmax(disease_logits, dim=-1)[0].cpu().tolist()
    class_names = ["HC", "AD/MCI"] if combine_mci_ad else ["HC", "AD", "MCI"]
    prediction = int(torch.argmax(disease_logits, dim=-1).item())
    report = {
        "input": str(Path(args.input).resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "prediction": class_names[prediction],
        "probabilities": dict(zip(class_names, disease_probabilities)),
    }
    if language_logits is not None:
        probabilities = torch.softmax(language_logits, dim=-1)[0].cpu().tolist()
        report["language_probabilities"] = {
            language.name: probabilities[language.value] for language in Language
        }
    if attention is not None:
        weights = attention.detach().cpu().reshape(-1).tolist()
        report["attention"] = [
            {"start_seconds": float(start), "end_seconds": float(end), "weight": float(weight)}
            for (start, end), weight in zip(ranges.tolist(), weights)
        ]

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output_json:
        Path(args.output_json).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
