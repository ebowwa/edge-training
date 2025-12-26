"""
Roboflow Dataset Downloader

Downloads YOLO-format datasets from Roboflow export links.

Roboflow provides download links like:
https://app.roboflow.com/ds/your-key?style=zip

This module downloads, extracts, and prepares the dataset for training.
"""

import logging
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
import yaml

logger = logging.getLogger(__name__)


class RoboflowDatasetDownloader:
    """
    Download and prepare Roboflow datasets for YOLO training.

    Usage:
        downloader = RoboflowDatasetDownloader()

        # From download URL
        yaml_path = downloader.download_from_url(
            url="https://app.roboflow.com/ds/abc123?style=zip",
            output_dir="datasets/roboflow"
        )

        # Or from project/workspace + API key
        yaml_path = downloader.download_from_api(
            workspace="my-workspace",
            project="my-project",
            version=1,
            api_key="rf_...",
            output_dir="datasets/roboflow"
        )
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize downloader.

        Args:
            cache_dir: Directory to cache downloads (default: ./datasets/cache)
        """
        self.cache_dir = Path(cache_dir or "datasets/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_from_url(
        self,
        url: str,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Download dataset from Roboflow export URL.

        Args:
            url: Roboflow download URL (from Export > Download > YAML)
                   Format: https://app.roboflow.com/ds/KEY?style=zip
                          or https://app.roboflow.com/ds/KEY?key=API_KEY
            output_dir: Where to extract dataset (default: datasets/roboflow)

        Returns:
            Path to data.yaml for training
        """
        output_path = Path(output_dir or "datasets/roboflow")
        output_path.mkdir(parents=True, exist_ok=True)

        # Extract key from URL
        # Format: https://app.roboflow.com/ds/abc123?style=zip
        #         or https://app.roboflow.com/ds/abc123?key=xyz
        if "/ds/" not in url:
            raise ValueError("Invalid Roboflow URL. Expected: https://app.roboflow.com/ds/KEY?style=zip")

        key = url.split("/ds/")[1].split("?")[0]
        zip_name = f"roboflow_{key}.zip"
        zip_path = self.cache_dir / zip_name

        # Ensure URL has style=zip parameter for proper download
        if "style=" not in url:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}style=zip"

        # Download if not cached
        if not zip_path.exists():
            logger.info(f"Downloading from Roboflow: {key}")
            self._download_file(url, zip_path)
        else:
            logger.info(f"Using cached: {zip_path}")

        # Extract
        logger.info(f"Extracting to {output_path}")
        return self._extract_dataset(zip_path, output_path)

    def download_from_api(
        self,
        workspace: str,
        project: str,
        version: int,
        api_key: str,
        model_format: str = "yolov8",
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Download dataset using Roboflow API (requires API key).

        Args:
            workspace: Roboflow workspace name
            project: Project name
            version: Dataset version
            api_key: Roboflow API key (rf_...)
            model_format: Export format (yolov8, yolov5, etc.)
            output_dir: Where to extract dataset

        Returns:
            Path to data.yaml for training
        """
        output_path = Path(output_dir or "datasets/roboflow")
        output_path.mkdir(parents=True, exist_ok=True)

        # Construct download URL
        url = f"https://app.roboflow.com/ds/{workspace}-{project}-{version}-{model_format}?api_key={api_key}"

        zip_name = f"roboflow_{workspace}_{project}_v{version}.zip"
        zip_path = self.cache_dir / zip_name

        # Download
        if not zip_path.exists():
            logger.info(f"Downloading {workspace}/{project} v{version}")
            self._download_file(url, zip_path)
        else:
            logger.info(f"Using cached: {zip_path}")

        # Extract
        return self._extract_dataset(zip_path, output_path)

    def _download_file(self, url: str, dest: Path):
        """Download file with progress."""
        with httpx.stream("GET", url, timeout=60.0, follow_redirects=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))

            with open(dest, "wb") as f:
                downloaded = 0
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024) == 0 or downloaded == total_size:
                            logger.info(f"  Downloaded: {downloaded / (1024*1024):.1f}MB ({pct:.0f}%)")

        logger.info(f"Downloaded: {dest}")

    def _extract_dataset(self, zip_path: Path, output_dir: Path) -> str:
        """
        Extract zip and return path to data.yaml.

        Roboflow exports contain:
        - data.yaml (or roboflow.config.yaml)
        - train/images/, train/labels/
        - valid/images/, valid/labels/
        - test/images/, test/labels/ (optional)
        """
        # Extract
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)

        # Find data.yaml
        yaml_path = self._find_data_yaml(output_dir)

        if yaml_path is None:
            # Try to create from structure
            yaml_path = self._create_data_yaml(output_dir)

        logger.info(f"Dataset ready: {yaml_path}")
        return str(yaml_path)

    def _find_data_yaml(self, base_dir: Path) -> Optional[Path]:
        """Find data.yaml in extracted directory."""
        candidates = [
            base_dir / "data.yaml",
            base_dir / "roboflow.config.yaml",
        ]

        # Also search one level deep
        for child in base_dir.iterdir():
            if child.is_dir():
                candidates.extend([
                    child / "data.yaml",
                    child / "roboflow.config.yaml",
                ])

        for path in candidates:
            if path.exists():
                return path

        return None

    def _create_data_yaml(self, base_dir: Path) -> Path:
        """
        Create data.yaml from detected dataset structure.

        Roboflow YOLO exports usually have:
        - train/images, train/labels
        - valid/images, valid/labels
        - test/images, test/labels
        """
        # Detect structure
        train_images = self._find_dir(base_dir, "train", "images")
        train_labels = self._find_dir(base_dir, "train", "labels")
        val_images = self._find_dir(base_dir, "valid", "images") or self._find_dir(base_dir, "val", "images")
        val_labels = self._find_dir(base_dir, "valid", "labels") or self._find_dir(base_dir, "val", "labels")
        test_images = self._find_dir(base_dir, "test", "images")
        test_labels = self._find_dir(base_dir, "test", "labels")

        # Count classes from a label file
        nc, names = self._detect_classes(train_labels)

        # Create data.yaml
        yaml_path = base_dir / "data.yaml"
        data = {
            "path": str(base_dir),
            "train": str(train_images.relative_to(base_dir)) if train_images else "train/images",
            "val": str(val_images.relative_to(base_dir)) if val_images else "valid/images",
            "test": str(test_images.relative_to(base_dir)) if test_images else "test/images",
            "nc": nc,
            "names": names,
        }

        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

        logger.info(f"Created data.yaml with {nc} classes: {names}")
        return yaml_path

    def _find_dir(self, base: Path, *parts) -> Optional[Path]:
        """Find directory matching pattern."""
        for child in base.iterdir():
            if child.is_dir() and child.name.lower() in [p.lower() for p in parts]:
                return child
        # Also check nested
        for child in base.rglob("*"):
            if child.is_dir() and child.name.lower() == parts[-1].lower():
                if all(p.lower() in str(child).lower() for p in parts):
                    return child
        return None

    def _detect_classes(self, labels_dir: Optional[Path]) -> Tuple[int, List[str]]:
        """Detect number of classes from label files."""
        if not labels_dir or not labels_dir.exists():
            return 1, ["class_0"]

        # Read first few label files to find max class index
        max_class = 0
        count = 0

        for label_file in labels_dir.iterdir():
            if label_file.suffix == ".txt" and count < 10:
                try:
                    with open(label_file) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 1:
                                class_id = int(parts[0])
                                max_class = max(max_class, class_id)
                    count += 1
                except Exception:
                    pass

        nc = max_class + 1
        names = [f"class_{i}" for i in range(nc)]
        return nc, names

    def get_dataset_info(self, yaml_path: str) -> Dict:
        """Get dataset statistics from yaml path."""
        yaml_path = Path(yaml_path)

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        # Count images
        info = {
            "nc": data.get("nc", 0),
            "names": data.get("names", []),
        }

        for split in ["train", "val", "test"]:
            split_path = Path(data.get("path", ".")) / data.get(split, "")
            if split_path.exists():
                images_dir = split_path / "images"
                if images_dir.exists():
                    info[f"{split}_images"] = len(list(images_dir.glob("*.jpg")))

        return info


def download_roboflow_dataset(
    url: str,
    output_dir: str = "datasets/roboflow",
) -> str:
    """
    Quick helper to download Roboflow dataset.

    Args:
        url: Roboflow export URL
        output_dir: Where to extract

    Returns:
        Path to data.yaml ready for YOLO training

    Example:
        yaml_path = download_roboflow_dataset(
            "https://app.roboflow.com/ds/abc123?style=zip"
        )
        print(f"Dataset ready: {yaml_path}")
    """
    downloader = RoboflowDatasetDownloader()
    return downloader.download_from_url(url, output_dir)
