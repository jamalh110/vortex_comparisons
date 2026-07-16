#!/usr/bin/env python3
"""Build the `/mydata` layout used by pipeline1.

Run on one GPU node after installing FLMR and Faiss, then distribute `/mydata`
with `cluster.sh sync-data`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path


HF_DATASET = "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR"
IMAGE_URLS = {
    "google-landmark.tar": (
        "https://huggingface.co/datasets/BByrneLab/M2KR_Images/resolve/main/"
        "EVQA/google-landmark.tar"
    ),
    "inat.zip": (
        "https://huggingface.co/datasets/BByrneLab/M2KR_Images/resolve/main/"
        "EVQA/inat.zip"
    ),
}


def download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size:
        print(f"skip existing {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    print(f"download {url} -> {destination}")
    urllib.request.urlretrieve(url, partial)
    partial.replace(destination)


def download_models(root: Path) -> None:
    from huggingface_hub import snapshot_download

    for repo, directory in (
        ("LinWeizheDragon/PreFLMR_ViT-L", root / "PreFLMR_ViT-L"),
        ("openai/clip-vit-large-patch14", root / "clip-vit-large-patch14"),
    ):
        if (directory / "config.json").exists():
            print(f"skip existing {directory}")
            continue
        print(f"download model {repo}")
        snapshot_download(
            repo_id=repo,
            local_dir=str(directory),
            local_dir_use_symlinks=False,
        )


def save_dataset_tables(evqa_root: Path) -> None:
    from datasets import load_dataset

    data = load_dataset(HF_DATASET, "EVQA_data")
    data_dir = evqa_root / "EVQA_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "train": "train-00000-of-00001.parquet",
        "test": "test-00000-of-00001-2.parquet",
        "valid": "valid-00000-of-00001.parquet",
        "validation": "valid-00000-of-00001.parquet",
    }
    for split, filename in names.items():
        if split in data:
            path = data_dir / filename
            if not path.exists():
                print(f"write {path}")
                data[split].to_parquet(str(path))

    passages = load_dataset(HF_DATASET, "EVQA_passages")
    passage_dir = evqa_root / "EVQA_passages"
    passage_dir.mkdir(parents=True, exist_ok=True)
    for split, filename in (
        ("train_passages", "train_passages-00000-of-00001.parquet"),
        ("test_passages", "test_passages-00000-of-00001.parquet"),
        ("valid_passages", "valid_passages-00000-of-00001.parquet"),
    ):
        if split in passages:
            path = passage_dir / filename
            if not path.exists():
                print(f"write {path}")
                passages[split].to_parquet(str(path))


def download_images(evqa_root: Path) -> None:
    for filename, url in IMAGE_URLS.items():
        archive = evqa_root / filename
        download(url, archive)
        if archive.suffix == ".tar":
            output = evqa_root / "google-landmark"
            if not output.exists() or not any(output.iterdir()):
                output.mkdir(parents=True, exist_ok=True)
                with tarfile.open(archive) as handle:
                    handle.extractall(output)
        else:
            output = evqa_root / "inat"
            if not output.exists() or not any(output.iterdir()):
                output.mkdir(parents=True, exist_ok=True)
                try:
                    with zipfile.ZipFile(archive) as handle:
                        handle.extractall(output)
                except zipfile.BadZipFile as exc:
                    raise RuntimeError(
                        f"{archive} is corrupt. Redownload it or repair it with "
                        "`zip -FF inat.zip --out inat_fixed.zip`."
                    ) from exc

    nested = evqa_root / "inat/inat"
    if nested.is_dir():
        for child in nested.iterdir():
            destination = nested.parent / child.name
            if not destination.exists():
                shutil.move(str(child), destination)
        nested.rmdir()


def extract_step_models(data_root: Path) -> None:
    import torch
    from flmr import (
        FLMRConfig,
        FLMRContextEncoderTokenizer,
        FLMRModelForRetrieval,
        FLMRQueryEncoderTokenizer,
    )

    checkpoint = data_root / "PreFLMR_ViT-L"
    output = data_root / "EVQA/models"
    output.mkdir(parents=True, exist_ok=True)
    required = [
        output / "models_step_A_query_text_encoder.pt",
        output / "models_step_A_query_text_linear.pt",
        output / "models_step_B_vision_encoder.pt",
        output / "models_step_B_vision_projection.pt",
        output / "models_step_C_transformer_mapping_input_linear.pt",
        output / "models_step_D_transformer_mapping.pt",
        output / "models_step_D_transformer_mapping_output.pt",
    ]
    if all(path.exists() for path in required):
        print("skip existing extracted step models")
        return

    config = FLMRConfig.from_pretrained(checkpoint)
    query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(
        checkpoint,
        text_config=config.text_config,
        subfolder="query_tokenizer",
    )
    context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(
        checkpoint,
        text_config=config.text_config,
        subfolder="context_tokenizer",
    )
    model = FLMRModelForRetrieval.from_pretrained(
        checkpoint,
        query_tokenizer=query_tokenizer,
        context_tokenizer=context_tokenizer,
    )
    mapping = {
        required[0]: model.query_text_encoder.state_dict(),
        required[1]: model.query_text_encoder_linear.state_dict(),
        required[2]: model.query_vision_encoder.state_dict(),
        required[3]: model.query_vision_projection.state_dict(),
        required[4]: model.transformer_mapping_input_linear.state_dict(),
        required[5]: model.transformer_mapping_network.state_dict(),
        required[6]: model.transformer_mapping_output_linear.state_dict(),
    }
    for path, state in mapping.items():
        torch.save(state, path)
        print(f"wrote {path}")


def build_index(data_root: Path, overwrite: bool) -> None:
    from datasets import load_dataset
    from flmr import index_custom_collection

    parquet = (
        data_root
        / "EVQA/EVQA_passages/train_passages-00000-of-00001.parquet"
    )
    index_root = data_root / "EVQA/index"
    target = (
        index_root
        / "EVQA_train_split/indexes/EVQA_PreFLMR_ViT-L.nbits=8"
    )
    if target.exists() and not overwrite:
        print(f"skip existing index {target}")
        return

    dataset = load_dataset(
        "parquet",
        data_files={"train": str(parquet)},
    )["train"]
    index_custom_collection(
        custom_collection=dataset["passage_content"],
        model=str(data_root / "PreFLMR_ViT-L"),
        index_root_path=str(index_root),
        index_experiment_name="EVQA_train_split",
        index_name="EVQA_PreFLMR_ViT-L",
        nbits=8,
        doc_maxlen=512,
        overwrite=True,
        use_gpu=True,
        indexing_batch_size=64,
        model_temp_folder="/tmp/flmr_index_tmp",
        nranks=1,
    )
    print(f"index ready under {index_root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", "/mydata")),
    )
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--overwrite-index", action="store_true")
    args = parser.parse_args()

    args.data_root.mkdir(parents=True, exist_ok=True)
    evqa_root = args.data_root / "EVQA"
    evqa_root.mkdir(parents=True, exist_ok=True)

    download_models(args.data_root)
    save_dataset_tables(evqa_root)
    if not args.skip_images:
        download_images(evqa_root)
    extract_step_models(args.data_root)
    if not args.skip_index:
        build_index(args.data_root, args.overwrite_index)

    os.chmod(args.data_root, 0o777)
    print("Data preparation complete.")


if __name__ == "__main__":
    main()

