import csv
import hashlib
import html
import io
import json
import random
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


BASE_DIR = Path("/home/santosh/Desktop/MorphTrace/data/casia")
REAL_DIR = BASE_DIR / "Au"
FAKE_DIR = BASE_DIR / "Tp"
MANIFEST = BASE_DIR / "au_download_manifest.csv"
SEARCH_STATE = BASE_DIR / "au_search_state.json"

TOTAL_REAL_IMAGES = 20_000
MAX_SEARCH_ATTEMPTS = max(500, TOTAL_REAL_IMAGES * 5)
MAX_DELAY_SECONDS = 0.5
MIN_DELAY_SECONDS = 0.2
MIN_IMAGE_SIZE = 64
MAX_IMAGE_WIDTH = 1024
RANDOM_SEED = 42

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "BalancedCameraPhotoDataset/1.0 (educational research)"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

CATEGORY_QUERIES = {
    "people": {
        "percentage": 40,
        "queries": (
            "portrait India",
            "selfie India",
            "India woman",
            "India man",
            "India people",
            "Indian portrait",
            "India family",
            "India traditional dress",
        ),
    },
    "india_documents": {
        "percentage": 10,
        "queries": (
            "India newspaper",
            "India newspaper page",
            "India historical book",
            "India manuscript",
            "India book",
            "India map",
            "India document",
        ),
    },
    "india_other": {
        "percentage": 10,
        "queries": (
            "India landscape",
            "India building",
            "Indian monument",
            "India market",
            "India village",
            "India street",
            "Indian street scene",
        ),
    },
    "crowds": {
        "percentage": 10,
        "queries": (
            "India crowd",
            "India festival crowd",
            "India market crowd",
            "India gathering",
            "India railway crowd",
        ),
    },
    "animals": {
        "percentage": 20,
        "queries": (
            "India wildlife",
            "India bird",
            "India tiger",
            "India elephant",
            "India animal",
            "wildlife animal",
        ),
    },
    "miscellaneous": {
        "percentage": 10,
        "queries": (
            "India flower",
            "India food",
            "India vehicle",
            "India object",
            "India nature",
        ),
    },
}
if sum(category["percentage"] for category in CATEGORY_QUERIES.values()) != 100:
    raise ValueError("Category percentages must add up to 100.")


def allocate_category_targets(total):
    exact_targets = {
        category: total * settings["percentage"] / 100
        for category, settings in CATEGORY_QUERIES.items()
    }
    targets = {
        category: int(target)
        for category, target in exact_targets.items()
    }
    remaining = total - sum(targets.values())
    remainder_order = sorted(
        exact_targets,
        key=lambda category: exact_targets[category] - targets[category],
        reverse=True,
    )
    for category in remainder_order[:remaining]:
        targets[category] += 1
    return targets

REUSABLE_LICENSES = {
    "cc0",
    "public domain",
    "cc by 2.0",
    "cc by 2.5",
    "cc by 3.0",
    "cc by 4.0",
    "cc by-sa 2.0",
    "cc by-sa 2.5",
    "cc by-sa 3.0",
    "cc by-sa 4.0",
}


def count_images(folder):
    return sum(
        1
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def calculate_hash(image):
    image = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(str(image.size).encode("utf-8"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def load_existing_hashes():
    hashes = set()
    for folder in (REAL_DIR, FAKE_DIR):
        for path in folder.iterdir():
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                with Image.open(path) as image:
                    hashes.add(calculate_hash(image.copy()))
            except (OSError, UnidentifiedImageError):
                continue
    return hashes


def count_existing_categories():
    counts = {category: 0 for category in CATEGORY_QUERIES}
    if not MANIFEST.exists():
        return counts

    counted_files = set()
    with MANIFEST.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            image_path = Path(row.get("filename", ""))
            category = row.get("category")
            if (
                row.get("label") == "Real"
                and category in counts
                and image_path.is_file()
                and str(image_path) not in counted_files
            ):
                counts[category] += 1
                counted_files.add(str(image_path))
    return counts


def count_existing_downloads():
    if not MANIFEST.exists():
        return 0

    with MANIFEST.open(newline="", encoding="utf-8") as csv_file:
        return len(
            {
                row.get("filename", "")
                for row in csv.DictReader(csv_file)
                if row.get("label") == "Real"
                and Path(row.get("filename", "")).is_file()
            }
        )


def load_existing_source_pages():
    if not MANIFEST.exists():
        return set()

    with MANIFEST.open(newline="", encoding="utf-8") as csv_file:
        return {
            row["source_page"]
            for row in csv.DictReader(csv_file)
            if row.get("label") == "Real"
            and Path(row.get("filename", "")).is_file()
            and row.get("source_page")
        }


def load_reserved_output_paths():
    if not MANIFEST.exists():
        return set()

    with MANIFEST.open(newline="", encoding="utf-8") as csv_file:
        return {
            row["filename"]
            for row in csv.DictReader(csv_file)
            if row.get("label") == "Real" and row.get("filename")
        }


def load_search_state(queries):
    if not SEARCH_STATE.exists():
        return (
            {query: None for query in queries},
            set(),
        )

    with SEARCH_STATE.open(encoding="utf-8") as state_file:
        state = json.load(state_file)

    offsets = {
        query: state.get("offsets", {}).get(query)
        for query in queries
    }
    exhausted_queries = set(state.get("exhausted_queries", ()))
    return offsets, exhausted_queries


def save_search_state(search_offsets, exhausted_queries):
    temporary = SEARCH_STATE.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as state_file:
        json.dump(
            {
                "offsets": search_offsets,
                "exhausted_queries": sorted(exhausted_queries),
            },
            state_file,
        )
    temporary.replace(SEARCH_STATE)


def commons_request(parameters):
    query = urllib.parse.urlencode(parameters)
    request = urllib.request.Request(
        f"{COMMONS_API}?{query}",
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def search_commons(topic, offset):
    parameters = {
        "action": "query",
        "generator": "search",
        "gsrsearch": topic,
        "gsrnamespace": 6,
        "gsrlimit": 50,
        "prop": "imageinfo",
        "iiprop": "url|metadata|extmetadata",
        "iiurlwidth": MAX_IMAGE_WIDTH,
        "format": "json",
        "formatversion": 2,
    }
    if offset is not None:
        parameters["gsroffset"] = offset

    response = commons_request(parameters)
    pages = response.get("query", {}).get("pages", [])
    next_offset = response.get("continue", {}).get("gsroffset")
    return pages, next_offset


def metadata_map(image_info):
    return {
        item["name"]: item.get("value")
        for item in image_info.get("metadata", [])
        if item.get("name")
    }


def extmetadata_value(image_info, name):
    value = image_info.get("extmetadata", {}).get(name, {}).get("value", "")
    return html.unescape(re.sub(r"<[^>]*>", "", str(value))).strip()


def parse_capture_date(value):
    if not value:
        return None

    match = re.match(r"^\s*(\d{4})[:\-](\d{2})[:\-](\d{2})", str(value))
    if not match:
        return None

    try:
        return datetime(*(int(part) for part in match.groups())).date()
    except ValueError:
        return None


def get_verified_candidate(page):
    image_info_list = page.get("imageinfo", [])
    if not image_info_list:
        return None

    image_info = image_info_list[0]
    metadata = metadata_map(image_info)
    capture_date = parse_capture_date(metadata.get("DateTimeOriginal"))
    if capture_date is None or capture_date.year >= 2013:
        return None

    license_name = extmetadata_value(image_info, "LicenseShortName")
    if license_name.casefold() not in REUSABLE_LICENSES:
        return None

    image_url = image_info.get("thumburl") or image_info.get("url")
    source_page = image_info.get("descriptionurl")
    if not image_url or not source_page:
        return None

    return {
        "image_url": image_url,
        "source_page": source_page,
        "capture_date": capture_date.isoformat(),
        "license": license_name,
        "license_url": extmetadata_value(image_info, "LicenseUrl"),
        "attribution": extmetadata_value(image_info, "Artist"),
        "camera_make": str(metadata.get("Make") or ""),
        "camera_model": str(metadata.get("Model") or ""),
        "title": page.get("title", ""),
        "description": extmetadata_value(image_info, "ImageDescription"),
        "categories": extmetadata_value(image_info, "Categories"),
    }


def matches_category(candidate, category):
    title_description = " ".join(
        (candidate["title"], candidate["description"])
    ).casefold()
    text = " ".join((title_description, candidate["categories"])).casefold()

    if category in {"india_documents", "india_other"} and not re.search(
        r"\bindia\b", text
    ):
        return False

    document_terms = (
        "document",
        "newspaper",
        "manuscript",
        "book",
        "map",
        "letter",
        "printed",
        "page",
    )
    has_document_term = any(
        re.search(rf"\b{re.escape(term)}\b", title_description)
        for term in document_terms
    )

    if category == "india_documents":
        unrelated_context = (
            "bag maker",
            "newspaper editor",
            "newspaper reading",
            "newspaper reader",
            "newspaper stall",
            "newspaper stand",
            "bookshop",
            "bookseller",
            "portrait",
        )
        return (
            has_document_term
            and not any(term in title_description for term in unrelated_context)
        )

    if category == "india_other" and has_document_term:
        return False

    required_terms = {
        "crowds": (
            "crowd",
            "crowded",
            "market",
            "festival",
            "gathering",
            "procession",
            "rally",
            "station",
        ),
        "animals": (
            "animal",
            "wildlife",
            "bird",
            "tiger",
            "elephant",
            "dog",
            "cat",
            "horse",
            "cattle",
            "species",
        ),
    }.get(category)

    if required_terms is None:
        return True

    has_required_term = any(
        re.search(rf"\b{re.escape(term)}\b", text)
        for term in required_terms
    )
    if category == "crowds" and re.search(
        r"\b(no|without|empty|crowdless)\s+crowd\b", text
    ):
        return False
    return has_required_term


def download_image(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        contents = response.read()
    with Image.open(io.BytesIO(contents)) as image:
        image.load()
        return image.convert("RGB")


def contains_face(image, detector):
    rgb_image = np.asarray(image.convert("RGB"))
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    return any(
        len(cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5))
        > 0
        for cascade in detector
    )


def build_face_detector():
    cascade_directory = Path(cv2.data.haarcascades)
    cascade_names = (
        "haarcascade_frontalface_default.xml",
        "haarcascade_profileface.xml",
    )
    detectors = []
    for name in cascade_names:
        cascade_path = cascade_directory / name
        detector = cv2.CascadeClassifier(str(cascade_path))
        if detector.empty():
            raise RuntimeError(f"Could not load the OpenCV face detector: {cascade_path}")
        detectors.append(detector)
    return detectors


def save_image(image, folder, prefix, reserved_output_paths):
    number = count_images(folder) + 1
    while True:
        output = folder / f"{prefix}_{number:03d}.png"
        if not output.exists() and str(output) not in reserved_output_paths:
            break
        number += 1

    temporary = output.with_suffix(".tmp")
    image.save(temporary, format="PNG")
    temporary.replace(output)
    reserved_output_paths.add(str(output))
    return output


def main():
    random.seed(RANDOM_SEED)
    if TOTAL_REAL_IMAGES < 1:
        raise ValueError("TOTAL_REAL_IMAGES must be at least 1.")
    category_targets = allocate_category_targets(TOTAL_REAL_IMAGES)
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    FAKE_DIR.mkdir(parents=True, exist_ok=True)

    counts = count_existing_categories()
    total_existing = count_existing_downloads()

    if total_existing >= TOTAL_REAL_IMAGES:
        return

    face_detector = build_face_detector()
    known_hashes = load_existing_hashes()
    manifest_exists = MANIFEST.exists()
    saved = 0
    seen_page_ids = set()
    used_source_pages = load_existing_source_pages()
    reserved_output_paths = load_reserved_output_paths()
    query_names = {
        query
        for category in CATEGORY_QUERIES.values()
        for query in category["queries"]
    }
    search_offsets, exhausted_queries = load_search_state(query_names)

    fields = [
        "filename",
        "label",
        "category",
        "dataset",
        "title",
        "source_page",
        "image_url",
        "capture_date",
        "license",
        "license_url",
        "attribution",
        "camera_make",
        "camera_model",
        "sha256",
    ]

    with MANIFEST.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        if not manifest_exists:
            writer.writeheader()

        with tqdm(
            total=TOTAL_REAL_IMAGES,
            initial=min(total_existing, TOTAL_REAL_IMAGES),
            desc="Images",
            unit="image",
        ) as progress:
            for _ in range(MAX_SEARCH_ATTEMPTS):
                if total_existing + saved >= TOTAL_REAL_IMAGES:
                    break

                available_queries = [
                    (category, query)
                    for category in CATEGORY_QUERIES
                    if counts[category] < category_targets[category]
                    for query in CATEGORY_QUERIES[category]["queries"]
                    if query not in exhausted_queries
                ]
                if not available_queries:
                    break

                category, topic = random.choice(available_queries)
                try:
                    pages, next_offset = search_commons(
                        topic,
                        search_offsets[topic],
                    )
                except (OSError, TimeoutError, ValueError) as error:
                    tqdm.write(f"Search request failed: {error}")
                    time.sleep(random.uniform(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS))
                    continue

                time.sleep(random.uniform(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS))
                random.shuffle(pages)

                for page in pages:
                    if (
                        counts[category] >= category_targets[category]
                        or total_existing + saved >= TOTAL_REAL_IMAGES
                    ):
                        break

                    page_id = page.get("pageid")
                    if (
                        page_id is None
                        or page_id in seen_page_ids
                    ):
                        continue
                    seen_page_ids.add(page_id)

                    candidate = get_verified_candidate(page)
                    if candidate is None or not matches_category(candidate, category):
                        continue
                    if candidate["source_page"] in used_source_pages:
                        continue

                    try:
                        image = download_image(candidate["image_url"])
                    except (OSError, TimeoutError, UnidentifiedImageError) as error:
                        tqdm.write(f"Image download failed: {error}")
                        continue

                    if image.width < MIN_IMAGE_SIZE or image.height < MIN_IMAGE_SIZE:
                        continue

                    if category == "people" and not contains_face(image, face_detector):
                        continue

                    digest = calculate_hash(image)
                    if digest in known_hashes:
                        continue

                    output = save_image(
                        image,
                        REAL_DIR,
                        "real",
                        reserved_output_paths,
                    )
                    known_hashes.add(digest)
                    used_source_pages.add(candidate["source_page"])
                    counts[category] += 1
                    saved += 1

                    writer.writerow(
                        {
                            "filename": str(output),
                            "label": "Real",
                            "category": category,
                            "dataset": "Wikimedia Commons",
                            "title": candidate["title"],
                            "source_page": candidate["source_page"],
                            "image_url": candidate["image_url"],
                            "capture_date": candidate["capture_date"],
                            "license": candidate["license"],
                            "license_url": candidate["license_url"],
                            "attribution": candidate["attribution"],
                            "camera_make": candidate["camera_make"],
                            "camera_model": candidate["camera_model"],
                            "sha256": digest,
                        }
                    )
                    csv_file.flush()
                    progress.update(1)

                    if saved < TOTAL_REAL_IMAGES - total_existing:
                        time.sleep(random.uniform(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS))

                search_offsets[topic] = next_offset
                if next_offset is None:
                    exhausted_queries.add(topic)
                save_search_state(search_offsets, exhausted_queries)


if __name__ == "__main__":
    main()
