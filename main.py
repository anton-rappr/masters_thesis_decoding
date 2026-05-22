from dotenv import load_dotenv
import os
from pathlib import Path
from modeling.train import (
    train_searchlight,
    train_wholebrain,
    train_searchlight_within_participant,
    train_wholebrain_within_participant,
)
from testing.permutation_test import (
    permutation_test_searchlight,
    permutation_test_wholebrain,
    permutation_test_searchlight_within_participant,
    permutation_test_wholebrain_within_participant,
)
from datetime import datetime
import json
import logging
from typing import List, Dict, Tuple
import numpy as np


def create_experiment_logger(exp_name: str, log_path: Path) -> logging.Logger:
    """create and return a logger configured to write to a file and stdout.

    parameters:
    exp_name : str
        name used as logger namespace and for log file naming.
    log_path : Path
        path to the logfile to write logs to.

    returns:
    logging.Logger
        configured logger instance.
    """

    logger = logging.getLogger(f"experiment.{exp_name}")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def save_experiment_metadata(results_dir: Path, metadata: dict):
    """save experiment metadata dict as a json file in `results_dir`.

    parameters:
    results_dir : Path
        directory where `experiment_metadata.json` will be written.
    metadata : dict
        metadata dictionary serializable to json.
    """

    metadata_file = results_dir / "experiment_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)


def load_experiment_config(config_path: str = "config/experiment.json") -> List[Dict]:
    """load experiment configuration json from `config_path`.

    parameters:
    config_path : str
        path to the experiment json config file.

    returns:
    List[Dict]
        list of experiment configurations loaded from the file.
    """

    with open(config_path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def normalize_experiment_config(config: Dict) -> Tuple[str, List[str], int]:
    """validate and normalize a single experiment config dict.

    parameters:
    config : Dict
        raw experiment config dictionary loaded from json.

    returns:
    Tuple[str, List[str], int]
        tuple of (name, labels_to_use, radius) extracted from the config.
    """

    name = str(config.get("name", "unnamed_experiment"))
    labels_to_use = config.get("labels_to_use")
    if not isinstance(labels_to_use, list) or not all(
        isinstance(label, str) for label in labels_to_use
    ):
        raise ValueError(
            "each experiment config must contain labels_to_use as a list[str]."
        )
    radius = int(config.get("radius", 8))
    return name, labels_to_use, radius


def load_env_paths() -> tuple[Path, Path, Path, Path | None]:
    """load environment variables from .env and return key path values.

    returns:
    tuple[Path, Path, Path, Path | None]
        (betas_dir, results_path, mask_path, process_mask_path)
    """

    load_dotenv()

    betas_dir = Path(os.getenv("beta_dir", ""))
    results_path = Path(os.getenv("results_path", ""))
    mask_path = Path(os.getenv("mask_path", ""))
    process_mask_path = (
        Path(os.getenv("process_mask_path", ""))
        if os.getenv("process_mask_path")
        else None
    )
    return betas_dir, results_path, mask_path, process_mask_path


def run_wholebrain_experiments(
    config_path: str = "config/experiment.json",
    n_jobs: int = 1,
    subject: str | None = None,
):
    """run whole-brain experiments from the config file and save results.

    parameters:
    config_path : str
        path to the experiment config json.
    n_jobs : int
        number of parallel jobs for training.
    subject : str | None
        if provided, run within-participant analysis for this subject.
    """

    betas_dir, results_path, mask_path, process_mask_path = load_env_paths()
    config_array = load_experiment_config(config_path)

    for config in config_array:
        name, labels_to_use, _ = normalize_experiment_config(config)

        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        subject_suffix = f"_{subject}" if subject else ""
        exp_results_dir = (
            results_path / f"{name}_wholebrain{subject_suffix}_{timestamp}"
        )
        exp_results_dir.mkdir(parents=True, exist_ok=True)

        log_path = exp_results_dir / "experiment.log"
        logger = create_experiment_logger(name, log_path)

        run_metadata = {
            "analysis_type": (
                "wholebrain" if subject is None else "wholebrain_within_participant"
            ),
            "experiment_name": name,
            "labels_to_use": labels_to_use,
            "subject": subject,
            "betas_dir": str(betas_dir),
            "mask_path": str(mask_path),
            "process_mask_path": str(process_mask_path) if process_mask_path else None,
            "results_path": str(exp_results_dir),
            "start_time": datetime.now().isoformat(),
        }

        logger.info(
            f"starting {'within-participant ' if subject else ''}whole-brain experiment: {name}"
        )

        try:
            if subject is None:
                cv_scores, _coef_img = train_wholebrain(
                    betas_dir=betas_dir,
                    labels_to_use=labels_to_use,
                    results_path=exp_results_dir,
                    mask_path=mask_path,
                    n_jobs=n_jobs,
                    experiment_name=name,
                )
            else:
                cv_scores, _coef_img = train_wholebrain_within_participant(
                    betas_dir=betas_dir,
                    labels_to_use=labels_to_use,
                    results_path=exp_results_dir,
                    mask_path=mask_path,
                    subject_id=subject,
                    n_jobs=n_jobs,
                    experiment_name=name,
                )

            run_metadata["train_metadata"] = {
                "cv_accuracy_mean": float(cv_scores.mean()),
                "cv_accuracy_std": float(cv_scores.std()),
            }
            run_metadata["end_time"] = datetime.now().isoformat()
            start = datetime.fromisoformat(run_metadata["start_time"])
            end = datetime.fromisoformat(run_metadata["end_time"])
            run_metadata["duration_seconds"] = (end - start).total_seconds()

            save_experiment_metadata(exp_results_dir, run_metadata)
            logger.info("whole-brain experiment completed successfully")

        except Exception as e:
            logger.exception("whole-brain experiment failed")
            run_metadata["error"] = str(e)
            run_metadata["end_time"] = datetime.now().isoformat()
            save_experiment_metadata(exp_results_dir, run_metadata)
            raise


def run_searchlight_experiments(
    config_path: str = "config/experiment.json",
    n_jobs: int = 1,
    single_slice_only: bool = False,
    subject: str | None = None,
):
    """run searchlight experiments from the config file and save results.

    parameters:
    config_path : str
        path to the experiment config json.
    n_jobs : int
        number of parallel jobs for training.
    single_slice_only : bool
        if true, restrict analysis to a single slice.
    subject : str | None
        if provided, run within-participant analysis for this subject.
    """

    betas_dir, results_path, mask_path, process_mask_path = load_env_paths()
    config_array = load_experiment_config(config_path)

    for config in config_array:
        name, labels_to_use, radius = normalize_experiment_config(config)

        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        subject_suffix = f"_{subject}" if subject else ""
        exp_results_dir = (
            results_path / f"{name}_searchlight_r{radius}{subject_suffix}_{timestamp}"
        )
        exp_results_dir.mkdir(parents=True, exist_ok=True)

        log_path = exp_results_dir / "experiment.log"
        logger = create_experiment_logger(name, log_path)

        run_metadata = {
            "analysis_type": (
                "searchlight" if subject is None else "searchlight_within_participant"
            ),
            "experiment_name": name,
            "radius": radius,
            "labels_to_use": labels_to_use,
            "subject": subject,
            "betas_dir": str(betas_dir),
            "mask_path": str(mask_path),
            "process_mask_path": str(process_mask_path) if process_mask_path else None,
            "results_path": str(exp_results_dir),
            "start_time": datetime.now().isoformat(),
        }

        logger.info(
            f"starting {'within-participant ' if subject else ''}searchlight experiment: {name}"
        )

        try:
            if subject is None:
                _searchlight_obj, train_meta = train_searchlight(
                    betas_dir=betas_dir,
                    labels_to_use=labels_to_use,
                    results_path=exp_results_dir,
                    mask_path=mask_path,
                    radius=radius,
                    n_jobs=n_jobs,
                    process_mask_path=process_mask_path,
                    logger=logger,
                    one_slice_only=single_slice_only,
                )
            else:
                _searchlight_obj, train_meta = train_searchlight_within_participant(
                    betas_dir=betas_dir,
                    labels_to_use=labels_to_use,
                    results_path=exp_results_dir,
                    mask_path=mask_path,
                    subject_id=subject,
                    radius=radius,
                    n_jobs=n_jobs,
                    process_mask_path=process_mask_path,
                    logger=logger,
                    one_slice_only=single_slice_only,
                )

            run_metadata["train_metadata"] = train_meta
            run_metadata["end_time"] = datetime.now().isoformat()
            start = datetime.fromisoformat(run_metadata["start_time"])
            end = datetime.fromisoformat(run_metadata["end_time"])
            run_metadata["duration_seconds"] = (end - start).total_seconds()

            save_experiment_metadata(exp_results_dir, run_metadata)
            logger.info("searchlight experiment completed successfully")

        except Exception as e:
            logger.exception("searchlight experiment failed")
            run_metadata["error"] = str(e)
            run_metadata["end_time"] = datetime.now().isoformat()
            save_experiment_metadata(exp_results_dir, run_metadata)
            raise


def run_permutation_test_searchlight_experiments(
    config_path: str = "config/experiment.json",
    n_permutations: int = 100,
    n_jobs: int = 4,
    random_state: int | None = None,
):
    """run searchlight permutation tests for configs in the config file.

    parameters:
    config_path : str
        path to the experiment config json.
    n_permutations : int
        number of permutations to run per config.
    n_jobs : int
        number of parallel jobs for searchlight.
    random_state : int | None
        optional random seed for reproducibility.
    """

    betas_dir, results_path, mask_path, process_mask_path = load_env_paths()
    config_array = load_experiment_config(config_path)

    for config in config_array:
        name, labels_to_use, radius = normalize_experiment_config(config)

        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        exp_results_dir = (
            results_path / f"{name}_searchlight_r{radius}_permutation_{timestamp}"
        )
        exp_results_dir.mkdir(parents=True, exist_ok=True)

        log_path = exp_results_dir / "permutation_test.log"
        logger = create_experiment_logger(name, log_path)

        run_metadata = {
            "analysis_type": "searchlight_permutation",
            "experiment_name": name,
            "radius": radius,
            "labels_to_use": labels_to_use,
            "n_permutations": n_permutations,
            "results_path": str(exp_results_dir),
            "start_time": datetime.now().isoformat(),
        }

        logger.info(f"starting searchlight permutation test: {name}")

        try:
            top_accuracies, accuracy_maps = permutation_test_searchlight(
                betas_dir=betas_dir,
                labels_to_use=labels_to_use,
                mask_path=mask_path,
                results_path=exp_results_dir,
                n_permutations=n_permutations,
                radius=radius,
                n_jobs=n_jobs,
                process_mask_path=process_mask_path,
                random_state=random_state,
                logger=logger,
            )

            run_metadata["end_time"] = datetime.now().isoformat()
            start = datetime.fromisoformat(run_metadata["start_time"])
            end = datetime.fromisoformat(run_metadata["end_time"])
            run_metadata["duration_seconds"] = (end - start).total_seconds()
            run_metadata["summary"] = {
                "n_permutations_completed": len(top_accuracies),
                "mean_top_accuracy": float(np.mean(top_accuracies)),
                "std_top_accuracy": float(np.std(top_accuracies)),
            }

            save_experiment_metadata(exp_results_dir, run_metadata)
            logger.info("searchlight permutation test completed successfully")

        except Exception as e:
            logger.exception("searchlight permutation test failed")
            run_metadata["error"] = str(e)
            run_metadata["end_time"] = datetime.now().isoformat()
            save_experiment_metadata(exp_results_dir, run_metadata)
            raise


def run_permutation_test_wholebrain_experiments(
    config_path: str = "config/experiment.json",
    n_permutations: int = 100,
    n_jobs: int = 1,
    random_state: int = 42,
):
    """run whole-brain permutation tests for configs in the config file.

    parameters:
    config_path : str
        path to the experiment config json.
    n_permutations : int
        number of permutations to run per config.
    n_jobs : int
        number of parallel jobs for cross-validation.
    random_state : int
        random seed for reproducibility.
    """

    betas_dir, results_path, mask_path, _ = load_env_paths()
    config_array = load_experiment_config(config_path)

    for config in config_array:
        name, labels_to_use, _ = normalize_experiment_config(config)

        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        exp_results_dir = results_path / f"{name}_wholebrain_permutation_{timestamp}"
        exp_results_dir.mkdir(parents=True, exist_ok=True)

        log_path = exp_results_dir / "permutation_test.log"
        logger = create_experiment_logger(name, log_path)

        run_metadata = {
            "analysis_type": "wholebrain_permutation",
            "experiment_name": name,
            "labels_to_use": labels_to_use,
            "n_permutations": n_permutations,
            "results_path": str(exp_results_dir),
            "start_time": datetime.now().isoformat(),
        }

        logger.info(f"starting whole-brain permutation test: {name}")

        try:
            top_accuracies = permutation_test_wholebrain(
                betas_dir=betas_dir,
                labels_to_use=labels_to_use,
                mask_path=mask_path,
                results_path=exp_results_dir,
                n_permutations=n_permutations,
                n_jobs=n_jobs,
                random_state=random_state,
                logger=logger,
            )

            run_metadata["end_time"] = datetime.now().isoformat()
            start = datetime.fromisoformat(run_metadata["start_time"])
            end = datetime.fromisoformat(run_metadata["end_time"])
            run_metadata["duration_seconds"] = (end - start).total_seconds()
            run_metadata["summary"] = {
                "n_permutations_completed": len(top_accuracies),
                "mean_top_accuracy": float(np.mean(top_accuracies)),
                "std_top_accuracy": float(np.std(top_accuracies)),
            }

            save_experiment_metadata(exp_results_dir, run_metadata)
            logger.info("whole-brain permutation test completed successfully")

        except Exception as e:
            logger.exception("whole-brain permutation test failed")
            run_metadata["error"] = str(e)
            run_metadata["end_time"] = datetime.now().isoformat()
            save_experiment_metadata(exp_results_dir, run_metadata)
            raise


def run_permutation_test_searchlight_within_participant_experiments(
    config_path: str = "config/experiment.json",
    subject: str | None = None,
    n_permutations: int = 100,
    n_jobs: int = 4,
    random_state: int | None = None,
):
    """run within-participant searchlight permutation tests for configs.

    parameters:
    config_path : str
        path to the experiment config json.
    subject : str | None
        subject id to analyze (required).
    n_permutations : int
        number of permutations to run.
    n_jobs : int
        number of parallel jobs for searchlight.
    random_state : int | None
        optional random seed for reproducibility.
    """
    if subject is None:
        raise Exception(
            "Error: subject parameter required for within-participant analysis"
        )
        return

    betas_dir, results_path, mask_path, process_mask_path = load_env_paths()
    config_array = load_experiment_config(config_path)

    for config in config_array:
        name, labels_to_use, radius = normalize_experiment_config(config)

        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        exp_results_dir = (
            results_path
            / f"{name}_searchlight_r{radius}_{subject}_permutation_{timestamp}"
        )
        exp_results_dir.mkdir(parents=True, exist_ok=True)

        log_path = exp_results_dir / "permutation_test.log"
        logger = create_experiment_logger(name, log_path)

        run_metadata = {
            "analysis_type": "searchlight_permutation_within_participant",
            "experiment_name": name,
            "subject": subject,
            "radius": radius,
            "labels_to_use": labels_to_use,
            "n_permutations": n_permutations,
            "results_path": str(exp_results_dir),
            "start_time": datetime.now().isoformat(),
        }

        logger.info(
            f"starting searchlight permutation test for subject {subject}: {name}"
        )

        try:
            top_accuracies, accuracy_maps = (
                permutation_test_searchlight_within_participant(
                    betas_dir=betas_dir,
                    labels_to_use=labels_to_use,
                    mask_path=mask_path,
                    results_path=exp_results_dir,
                    subject_id=subject,
                    n_permutations=n_permutations,
                    radius=radius,
                    n_jobs=n_jobs,
                    process_mask_path=process_mask_path,
                    random_state=random_state,
                    logger=logger,
                )
            )

            run_metadata["end_time"] = datetime.now().isoformat()
            start = datetime.fromisoformat(run_metadata["start_time"])
            end = datetime.fromisoformat(run_metadata["end_time"])
            run_metadata["duration_seconds"] = (end - start).total_seconds()
            run_metadata["summary"] = {
                "n_permutations_completed": len(top_accuracies),
                "mean_top_accuracy": float(np.mean(top_accuracies)),
                "std_top_accuracy": float(np.std(top_accuracies)),
            }

            save_experiment_metadata(exp_results_dir, run_metadata)
            logger.info("searchlight permutation test completed successfully")

        except Exception as e:
            logger.exception("searchlight permutation test failed")
            run_metadata["error"] = str(e)
            run_metadata["end_time"] = datetime.now().isoformat()
            save_experiment_metadata(exp_results_dir, run_metadata)
            raise


def run_permutation_test_wholebrain_within_participant_experiments(
    config_path: str = "config/experiment.json",
    subject: str | None = None,
    n_permutations: int = 100,
    n_jobs: int = 1,
    random_state: int | None = None,
):
    """run within-participant whole-brain permutation tests for configs.

    parameters:
    config_path : str
        path to the experiment config json.
    subject : str | None
        subject id to analyze (required).
    n_permutations : int
        number of permutations to run.
    n_jobs : int
        number of parallel jobs for cross-validation.
    random_state : int | None
        optional random seed for reproducibility.
    """
    if subject is None:
        raise Exception(
            "Error: subject parameter required for within-participant analysis"
        )

    betas_dir, results_path, mask_path, process_mask_path = load_env_paths()
    config_array = load_experiment_config(config_path)

    for config in config_array:
        name, labels_to_use, _ = normalize_experiment_config(config)

        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        exp_results_dir = (
            results_path / f"{name}_wholebrain_{subject}_permutation_{timestamp}"
        )
        exp_results_dir.mkdir(parents=True, exist_ok=True)

        log_path = exp_results_dir / "permutation_test.log"
        logger = create_experiment_logger(name, log_path)

        run_metadata = {
            "analysis_type": "wholebrain_permutation_within_participant",
            "experiment_name": name,
            "subject": subject,
            "labels_to_use": labels_to_use,
            "n_permutations": n_permutations,
            "results_path": str(exp_results_dir),
            "start_time": datetime.now().isoformat(),
        }

        logger.info(
            f"starting whole-brain permutation test for subject {subject}: {name}"
        )

        try:
            top_accuracies = permutation_test_wholebrain_within_participant(
                betas_dir=betas_dir,
                labels_to_use=labels_to_use,
                mask_path=mask_path,
                results_path=exp_results_dir,
                subject_id=subject,
                n_permutations=n_permutations,
                n_jobs=n_jobs,
                random_state=random_state,
                logger=logger,
            )

            run_metadata["end_time"] = datetime.now().isoformat()
            start = datetime.fromisoformat(run_metadata["start_time"])
            end = datetime.fromisoformat(run_metadata["end_time"])
            run_metadata["duration_seconds"] = (end - start).total_seconds()
            run_metadata["summary"] = {
                "n_permutations_completed": len(top_accuracies),
                "mean_top_accuracy": float(np.mean(top_accuracies)),
                "std_top_accuracy": float(np.std(top_accuracies)),
            }

            save_experiment_metadata(exp_results_dir, run_metadata)
            logger.info("whole-brain permutation test completed successfully")

        except Exception as e:
            logger.exception("whole-brain permutation test failed")
            run_metadata["error"] = str(e)
            run_metadata["end_time"] = datetime.now().isoformat()
            save_experiment_metadata(exp_results_dir, run_metadata)
            raise


def main():

    # run wholebrain experiments for four sample subjects
    random_state = 42
    subjs = ["S24", "S27", "S10", "S23"]
    for subj in subjs:
        print(f"running whole-brain experiment for subject {subj}")
        # example wholebrian within participant experiment
        run_wholebrain_experiments(
            config_path="config/experiment.json", n_jobs=1, subject=subj
        )

        # example wholebrain within participant permutation test
        run_permutation_test_wholebrain_within_participant_experiments(
            config_path="config/experiment.json",
            subject=subj,
            n_permutations=1000,
            n_jobs=1,
            random_state=random_state,
        )

        print(f"running searchlight experiment for subject {subj}")
        # example searchlight within participant analysis
        run_searchlight_experiments(
            config_path="config/experiment.json", n_jobs=-1, subject=subj
        )

        # example searchlight within participant permutation test
        run_permutation_test_searchlight_within_participant_experiments(
            config_path="config/experiment.json",
            subject=subj,
            n_permutations=1000,
            n_jobs=-1,
            random_state=random_state,
        )

    # example wholebrain across participant experiment
    run_wholebrain_experiments(
        config_path="config/experiment.json",
    )

    # example wholebrain across participant permutation test
    run_permutation_test_wholebrain_experiments(
        config_path="config/experiment.json",
        n_permutations=1000,
        n_jobs=1,
        random_state=random_state,
    )

    # example across participant searchlight analysis
    run_searchlight_experiments(
        config_path="config/experiment.json",
    )

    # example searchlight permutation test
    run_permutation_test_searchlight_experiments(
        config_path="config/experiment.json",
        n_permutations=100,
        n_jobs=-1,
        random_state=random_state,
    )


if __name__ == "__main__":
    main()
