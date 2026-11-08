import csv
import hashlib
import io
import logging
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import disable_progress_bars
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


AI_ART_REPOSITORY = "JoyCN/ai-generated-ecommerce-images"
AI_ART_REVISION = "0ede5c4c360e58a1010bd63169abee81e374684d"
DIFFUSIONDB_REPOSITORY = "poloclub/diffusiondb"
DIFFUSIONDB_REVISION = "fb620fbe49fa4420e0734bd9c0df11f51176b61f"
BASE_DIR = Path("/home/santosh/Desktop/MorphTrace/data/casia")
FAKE_DIR = BASE_DIR / "Tp"
MANIFEST = BASE_DIR / "tp_download_manifest.csv"
TARGET_FAKE_IMAGES = 20_000
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
OUTPUT_EXTENSION = ".jpg"
MIN_IMAGE_SIZE = 64


def calculate_hash(image):
    image = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(str(image.size).encode("utf-8"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def count_downloaded_images():
    if not MANIFEST.exists():
        return 0

    with MANIFEST.open(newline="", encoding="utf-8") as csv_file:
        return len(
            {
                row.get("filename", "")
                for row in csv.DictReader(csv_file)
                if row.get("label") == "Fake"
                and Path(row.get("filename", "")).is_file()
            }
        )


def load_existing_hashes():
    hashes = set()
    for folder in (BASE_DIR / "Au", FAKE_DIR):
        for path in folder.iterdir():
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                with Image.open(path) as image:
                    hashes.add(calculate_hash(image.copy()))
            except (OSError, UnidentifiedImageError) as error:
                tqdm.write(f"Could not read {path.name}: {error}")
    return hashes


def load_existing_source_files():
    if not MANIFEST.exists():
        return set()

    source_files = set()
    with MANIFEST.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            if (
                row.get("label") == "Fake"
                and Path(row.get("filename", "")).is_file()
                and row.get("source_file")
            ):
                source_files.add(row["source_file"])
    return source_files


def load_reserved_output_paths():
    if not MANIFEST.exists():
        return set()

    with MANIFEST.open(newline="", encoding="utf-8") as csv_file:
        return {
            row["filename"]
            for row in csv.DictReader(csv_file)
            if row.get("label") == "Fake" and row.get("filename")
        }


def verify_cc0_dataset(api, repository, revision):
    info = api.dataset_info(repository, revision=revision)
    licenses = (info.card_data or {}).get("license", [])
    if isinstance(licenses, str):
        licenses = [licenses]
    if info.gated or "cc0-1.0" not in licenses:
        raise RuntimeError(
            f"Expected an ungated CC0 dataset for {repository}; "
            f"got gated={info.gated}, license={licenses!r}."
        )


def save_ai_image(image, source, source_url, repository, license_name, hashes,
                  seen_source_files, reserved_output_paths, writer,
                  manifest_file, progress, downloaded_count):
    if image.width < MIN_IMAGE_SIZE or image.height < MIN_IMAGE_SIZE:
        return downloaded_count

    encoded_image = io.BytesIO()
    image.convert("RGB").save(
        encoded_image,
        format="JPEG",
        quality=95,
        optimize=True,
    )
    image_bytes = encoded_image.getvalue()
    with Image.open(io.BytesIO(image_bytes)) as saved_image:
        digest = calculate_hash(saved_image)
    if digest in hashes:
        return downloaded_count

    number = downloaded_count + 1
    output = FAKE_DIR / f"tp_{number:06d}{OUTPUT_EXTENSION}"
    while output.exists() or str(output) in reserved_output_paths:
        number += 1
        output = FAKE_DIR / f"tp_{number:06d}{OUTPUT_EXTENSION}"
    temporary = output.with_suffix(".tmp")
    temporary.write_bytes(image_bytes)
    temporary.replace(output)

    hashes.add(digest)
    seen_source_files.add(source)
    reserved_output_paths.add(str(output))
    writer.writerow(
        {
            "filename": str(output),
            "label": "Fake",
            "dataset": repository,
            "source_file": source,
            "source_url": source_url,
            "license": license_name,
            "sha256": digest,
        }
    )
    manifest_file.flush()
    progress.update(1)
    return number


def main():
    FAKE_DIR.mkdir(parents=True, exist_ok=True)
    downloaded_count = count_downloaded_images()
    if downloaded_count >= TARGET_FAKE_IMAGES:
        return

    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    disable_progress_bars()

    api = HfApi()
    verify_cc0_dataset(api, AI_ART_REPOSITORY, AI_ART_REVISION)
    verify_cc0_dataset(api, DIFFUSIONDB_REPOSITORY, DIFFUSIONDB_REVISION)

    ai_art_files = api.list_repo_files(
        AI_ART_REPOSITORY,
        repo_type="dataset",
        revision=AI_ART_REVISION,
    )
    ai_art_images = [
        path
        for path in ai_art_files
        if Path(path).suffix.lower() in IMAGE_EXTENSIONS
    ]
    diffusiondb_files = api.list_repo_files(
        DIFFUSIONDB_REPOSITORY,
        repo_type="dataset",
        revision=DIFFUSIONDB_REVISION,
    )
    diffusiondb_archives = sorted(
        path
        for path in diffusiondb_files
        if path.startswith("diffusiondb-large-part-1/")
        and path.endswith(".zip")
    )
    if not ai_art_images or not diffusiondb_archives:
        raise RuntimeError("The configured AI image sources contain no usable files.")

    manifest_has_header = MANIFEST.exists() and MANIFEST.stat().st_size > 0
    fields = [
        "filename",
        "label",
        "dataset",
        "source_file",
        "source_url",
        "license",
        "sha256",
    ]
    hashes = load_existing_hashes()
    seen_source_files = load_existing_source_files()
    reserved_output_paths = load_reserved_output_paths()

    with MANIFEST.open("a", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fields)
        if not manifest_has_header:
            writer.writeheader()

        with tqdm(
            total=TARGET_FAKE_IMAGES,
            initial=downloaded_count,
            desc="Tp images",
            unit="image",
        ) as progress:
            source_directory = Path(
                snapshot_download(
                    repo_id=AI_ART_REPOSITORY,
                    repo_type="dataset",
                    revision=AI_ART_REVISION,
                    allow_patterns=ai_art_images,
                )
            )
            for source_file in ai_art_images:
                if downloaded_count >= TARGET_FAKE_IMAGES:
                    break
                if source_file in seen_source_files:
                    continue

                local_file = source_directory / source_file
                try:
                    with Image.open(local_file) as image:
                        downloaded_count = save_ai_image(
                            image,
                            source_file,
                            (
                                f"https://huggingface.co/datasets/{AI_ART_REPOSITORY}"
                                f"/blob/{AI_ART_REVISION}/{source_file}"
                            ),
                            AI_ART_REPOSITORY,
                            "CC0-1.0",
                            hashes,
                            seen_source_files,
                            reserved_output_paths,
                            writer,
                            manifest_file,
                            progress,
                            downloaded_count,
                        )
                except (OSError, UnidentifiedImageError) as error:
                    tqdm.write(f"Could not process {source_file}: {error}")

            for archive_file in diffusiondb_archives:
                if downloaded_count >= TARGET_FAKE_IMAGES:
                    break

                archive_path = hf_hub_download(
                    repo_id=DIFFUSIONDB_REPOSITORY,
                    filename=archive_file,
                    repo_type="dataset",
                    revision=DIFFUSIONDB_REVISION,
                )
                try:
                    with zipfile.ZipFile(archive_path) as archive:
                        for item in archive.infolist():
                            if downloaded_count >= TARGET_FAKE_IMAGES:
                                break
                            if (
                                item.is_dir()
                                or Path(item.filename).suffix.lower()
                                not in IMAGE_EXTENSIONS
                            ):
                                continue

                            source_file = f"{archive_file}!/{item.filename}"
                            if source_file in seen_source_files:
                                continue
                            try:
                                with Image.open(io.BytesIO(archive.read(item))) as image:
                                    downloaded_count = save_ai_image(
                                        image,
                                        source_file,
                                        (
                                            f"https://huggingface.co/datasets/"
                                            f"{DIFFUSIONDB_REPOSITORY}/blob/"
                                            f"{DIFFUSIONDB_REVISION}/{archive_file}"
                                        ),
                                        DIFFUSIONDB_REPOSITORY,
                                        "CC0-1.0",
                                        hashes,
                                        seen_source_files,
                                        reserved_output_paths,
                                        writer,
                                        manifest_file,
                                        progress,
                                        downloaded_count,
                                    )
                            except (OSError, UnidentifiedImageError) as error:
                                tqdm.write(f"Could not process {source_file}: {error}")
                except (OSError, zipfile.BadZipFile) as error:
                    tqdm.write(f"Could not read {archive_file}: {error}")

            if downloaded_count < TARGET_FAKE_IMAGES:
                raise RuntimeError(
                    f"Only {downloaded_count} source-verified Tp images are "
                    f"available; {TARGET_FAKE_IMAGES} are required."
                )


if __name__ == "__main__":
    main()
