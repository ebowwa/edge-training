"""
Dataset Service for YOLO Training.
Handles dataset download, structure detection, and YAML configuration creation.
"""

import logging
import os
from typing import Dict, Optional, Tuple

import kagglehub
import yaml

from .config import DatasetConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class DatasetService:
    """Service for dataset operations."""

    @staticmethod
    def download(config: DatasetConfig) -> str:
        """
        Download the Kaggle dataset.

        Args:
            config: Dataset configuration with handle and metadata.

        Returns:
            Path to downloaded dataset.

        Raises:
            Exception: If download fails.
        """
        try:
            path = kagglehub.dataset_download(config.dataset_handle)
            logging.info(f"Downloaded dataset to: {path}")
            return path
        except Exception as e:
            logging.error(f"Failed to download dataset: {e}")
            raise

    @staticmethod
    def detect_structure(dataset_path: str) -> Tuple[Dict[str, str], str]:
        """
        Detect train/val/test images and labels paths in the dataset.

        Args:
            dataset_path: Path to the dataset root.

        Returns:
            Tuple of (paths dict, effective dataset path).
        """
        paths = {}
        subdirs = [
            d for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ]
        logging.info(f"Detected subdirs in dataset: {subdirs}")
        splits = ['train', 'valid', 'test']

        # Check top level
        for split in splits:
            key_split = 'val' if split == 'valid' else split
            if split in subdirs:
                split_dir = os.path.join(dataset_path, split)
                images_dir = os.path.join(split_dir, 'images')
                labels_dir = os.path.join(split_dir, 'labels')
                if os.path.exists(images_dir):
                    paths[f'{key_split}_images'] = images_dir
                if os.path.exists(labels_dir):
                    paths[f'{key_split}_labels'] = labels_dir

        if not paths:
            # Check one level deeper
            for subdir in subdirs:
                sub_path = os.path.join(dataset_path, subdir)
                if os.path.isdir(sub_path):
                    inner_subdirs = [
                        d for d in os.listdir(sub_path)
                        if os.path.isdir(os.path.join(sub_path, d))
                    ]
                    logging.info(
                        f"Checking inner subdirs in {subdir}: {inner_subdirs}"
                    )
                    for split in splits:
                        key_split = 'val' if split == 'valid' else split
                        if split in inner_subdirs:
                            split_dir = os.path.join(sub_path, split)
                            images_dir = os.path.join(split_dir, 'images')
                            labels_dir = os.path.join(split_dir, 'labels')
                            if os.path.exists(images_dir):
                                paths[f'{key_split}_images'] = images_dir
                            if os.path.exists(labels_dir):
                                paths[f'{key_split}_labels'] = labels_dir
                    if paths:
                        dataset_path = sub_path
                        break

            if not paths:
                # Check two levels deeper
                for subdir in subdirs:
                    sub_path = os.path.join(dataset_path, subdir)
                    if os.path.isdir(sub_path):
                        inner_subdirs = [
                            d for d in os.listdir(sub_path)
                            if os.path.isdir(os.path.join(sub_path, d))
                        ]
                        for inner_subdir in inner_subdirs:
                            inner_path = os.path.join(sub_path, inner_subdir)
                            if os.path.isdir(inner_path):
                                deepest_subdirs = [
                                    d for d in os.listdir(inner_path)
                                    if os.path.isdir(os.path.join(inner_path, d))
                                ]
                                logging.info(
                                    f"Checking deepest subdirs in "
                                    f"{inner_subdir}: {deepest_subdirs}"
                                )
                                for split in splits:
                                    key_split = 'val' if split == 'valid' else split
                                    if split in deepest_subdirs:
                                        split_dir = os.path.join(inner_path, split)
                                        images_dir = os.path.join(split_dir, 'images')
                                        labels_dir = os.path.join(split_dir, 'labels')
                                        if os.path.exists(images_dir):
                                            paths[f'{key_split}_images'] = images_dir
                                        if os.path.exists(labels_dir):
                                            paths[f'{key_split}_labels'] = labels_dir
                                if paths:
                                    dataset_path = inner_path
                                    break
                        if paths:
                            break

        logging.info(f"Detected paths: {paths}")
        return paths, dataset_path

    @staticmethod
    def create_yaml(
        dataset_path: str,
        paths: Dict[str, str],
        nc: int,
        names: list,
        output_path: Optional[str] = None
    ) -> str:
        """
        Create the data.yaml file for YOLO training.

        Args:
            dataset_path: Root path of the dataset.
            paths: Dictionary of detected image/label paths.
            nc: Number of classes.
            names: List of class names.
            output_path: Optional custom output path for YAML.

        Returns:
            Path to created YAML file.
        """
        data_yaml = {
            "path": dataset_path,
            "train": (
                os.path.relpath(paths.get('train_images', ''), dataset_path)
                if 'train_images' in paths else ''
            ),
            "val": (
                os.path.relpath(paths.get('val_images', ''), dataset_path)
                if 'val_images' in paths else ''
            ),
            "test": (
                os.path.relpath(paths.get('test_images', ''), dataset_path)
                if 'test_images' in paths else ''
            ),
            "nc": nc,
            "names": names,
        }

        if output_path is None:
            output_path = f"{os.path.basename(dataset_path)}.yaml"

        with open(output_path, "w") as f:
            yaml.dump(data_yaml, f)

        logging.info(f"Created YAML config at: {output_path}")
        return output_path

    @classmethod
    def prepare(cls, config: DatasetConfig) -> str:
        """
        Full pipeline: download, detect structure, create YAML.

        Args:
            config: Dataset configuration.

        Returns:
            Path to created YAML file.

        Raises:
            ValueError: If no valid dataset structure found.
        """
        dataset_path = cls.download(config)
        paths, dataset_path = cls.detect_structure(dataset_path)

        if not paths:
            raise ValueError(
                "No standard train/val/test structure found in dataset"
            )

        yaml_path = cls.create_yaml(
            dataset_path, paths, config.nc, config.names
        )
        return yaml_path
