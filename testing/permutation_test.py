from nilearn.image import resample_to_img, load_img
from nilearn.maskers import NiftiMasker
from sklearn.model_selection import LeaveOneGroupOut, cross_validate
from sklearn.svm import LinearSVC
from nilearn.decoding import SearchLight
from pathlib import Path
from datetime import datetime
import time
import numpy as np
import json
import logging
from typing import List
from data.data_loading import load_all_participants, filter_by_emotions
from data.masking import apply_mask, get_process_mask
import matplotlib.pyplot as plt
from modeling.train import SVC_C, SVC_MAX_ITER, SVC_RANDOM_STATE, TIME_ID


def permutation_test_searchlight(
    betas_dir: Path,
    labels_to_use: List[str],
    mask_path: Path,
    results_path: Path,
    n_permutations: int = 100,
    radius: int = 8,
    n_jobs: int = 4,
    process_mask_path: Path | None = None,
    random_state: int = 42,
    logger: logging.Logger | None = None,
):
    """run searchlight permutation test by shuffling labels globally.

    parameters:
    betas_dir : Path
        directory containing each subjects nifti folder.
    labels_to_use : List[str]
        labels/emotions to evaluate.
    mask_path : Path
        brain mask path used for resampling/masking.
    results_path : Path
        directory where permutation outputs are written.
    n_permutations : int
        number of permutations to run.
    radius : int
        searchlight radius in mm.
    n_jobs : int
        number of parallel jobs.
    process_mask_path : Path | None
        optional process mask path to restrict searchlight.
    random_state : int
        random seed for reproducibility.
    logger : logging.Logger | None
        optional logger for progress messages.

    returns:
    np.ndarray, list
        array of top accuracies and list of searchlight score images.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)

    # Set random state for reproducibility
    rng = np.random.RandomState(random_state)

    results_path.mkdir(parents=True, exist_ok=True)

    # load all data
    logger.info(f"loading all data from {betas_dir}")
    all_data_df = load_all_participants(betas_dir)
    logger.info(f"loaded {len(all_data_df)} total beta entries")

    top_accuracies = []
    accuracy_maps = []

    logger.info(f"Running {n_permutations} permutation iterations...")

    for perm_idx in range(n_permutations):
        logger.info(f"Permutation {perm_idx+1}/{n_permutations}")

        # permute labels globally at all_data_df level
        permuted_df = all_data_df.copy()
        permuted_df["label"] = rng.permutation(permuted_df["label"].values)

        # filter by emotions
        try:
            filtered_imgs, filtered_labels, filtered_subjs, _ = filter_by_emotions(
                permuted_df, labels_to_use
            )
        except ValueError as e:
            logger.warning(
                f"perm {perm_idx+1} produced invalid label distribution: {e}"
            )
            continue

        if len(np.unique(filtered_labels)) < 2:
            logger.warning(
                f"perm {perm_idx+1} has less than 2 unique labels after filtering"
            )
            continue

        # mask images
        masked_imgs = apply_mask(filtered_imgs, mask_path)
        process_mask = get_process_mask(process_mask_path, masked_imgs)

        # run searchlight
        cv = LeaveOneGroupOut()
        searchlight_object = SearchLight(
            masked_imgs,
            process_mask_img=process_mask,
            radius=radius,
            n_jobs=n_jobs,
            verbose=0,
            cv=cv,
        )

        searchlight_object.fit(
            filtered_imgs, filtered_labels, groups=filtered_subjs
        )  # add groups parameter for LO-Group-O CV
        scores_img = searchlight_object.scores_img_
        scores_data = scores_img.get_fdata()

        # get max accuracy in brain
        mask = scores_data > 0
        if np.any(mask):
            max_acc = np.max(scores_data[mask])
        else:
            max_acc = 0.0

        top_accuracies.append(max_acc)
        accuracy_maps.append(scores_img)

        # Save individual permutation map
        perm_scores_path = results_path / f"perm_{perm_idx:04d}_scores.nii.gz"
        scores_img.to_filename(perm_scores_path)
        logger.debug(f"saved permutation {perm_idx+1} scores to {perm_scores_path}")

        # save results every 10 permutations
        if (perm_idx + 1) % 10 == 0:
            logger.info(
                f"completed {perm_idx+1} permutations. Mean top accuracy: {np.mean(top_accuracies):.4f}"
            )

    top_accuracies = np.array(top_accuracies)

    # save top accuracies list
    top_accuracies_path = results_path / "top_accuracies.npy"
    np.save(top_accuracies_path, top_accuracies)
    logger.info(f"Saved top accuracies to {top_accuracies_path}")

    # create and save histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(top_accuracies, bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Max Accuracy")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of Top Accuracies ({n_permutations} permutations)")
    ax.axvline(
        np.mean(top_accuracies),
        color="r",
        linestyle="--",
        label=f"Mean: {np.mean(top_accuracies):.4f}",
    )
    ax.legend()

    histogram_path = results_path / "top_accuracies_histogram.png"
    plt.savefig(histogram_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved histogram to {histogram_path}")

    # save summary
    summary = {
        "n_permutations": int(n_permutations),
        "labels_used": labels_to_use,
        "radius": radius,
        "top_accuracies": {
            "mean": float(np.mean(top_accuracies)),
            "std": float(np.std(top_accuracies)),
            "min": float(np.min(top_accuracies)),
            "max": float(np.max(top_accuracies)),
            "median": float(np.median(top_accuracies)),
        },
        "timestamp": datetime.now().isoformat(),
    }

    summary_path = results_path / "permutation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")

    logger.info(
        f"Permutation test completed. Mean top accuracy: {np.mean(top_accuracies):.4f} +/- {np.std(top_accuracies):.4f}"
    )

    return top_accuracies, accuracy_maps


def permutation_test_wholebrain(
    betas_dir: Path,
    labels_to_use: List[str],
    mask_path: Path,
    results_path: Path,
    n_permutations: int = 100,
    n_jobs: int = 1,
    random_state: int | None = None,
    logger: logging.Logger | None = None,
):
    """run whole-brain permutation test by shuffling labels globally.

    parameters:
    betas_dir : Path
        directory containing each subjects nifti folder.
    labels_to_use : List[str]
        labels/emotions to evaluate.
    mask_path : Path
        brain mask path used for resampling/masking.
    results_path : Path
        directory where permutation outputs are written.
    n_permutations : int
        number of permutations to run.
    n_jobs : int
        number of parallel jobs for cross-validation.
    random_state : int | None
        random seed for reproducibility.
    logger : logging.Logger | None
        optional logger for progress messages.

    returns:
    np.ndarray
        array of top fold accuracies for each permutation.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)

    # set random state for reproducibility
    rng = np.random.RandomState(random_state)

    # load all data once
    logger.info(f"loading all data from {betas_dir}")
    all_data_df = load_all_participants(betas_dir)
    logger.info(f"loaded {len(all_data_df)} total beta entries")

    # prepare mask and classifier once
    mask_img = load_img(mask_path)
    cv = LeaveOneGroupOut()

    top_accuracies = []

    logger.info(f"running {n_permutations} permutation iterations...")

    for perm_idx in range(n_permutations):
        logger.info(f"Permutation {perm_idx+1}/{n_permutations}")

        # permute labels globally at all_data_df level
        permuted_df = all_data_df.copy()
        permuted_df["label"] = rng.permutation(permuted_df["label"].values)

        # filter by emotions
        try:
            filtered_imgs, filtered_labels, filtered_subjs, _ = filter_by_emotions(
                permuted_df, labels_to_use
            )
        except ValueError as e:
            logger.warning(
                f"perm {perm_idx+1} produced invalid label distribution: {e}"
            )
            continue

        if len(np.unique(filtered_labels)) < 2:
            logger.warning(
                f"perm {perm_idx+1} has less than 2 unique labels after filtering"
            )
            continue

        # prepare data
        resampled_mask = resample_to_img(
            mask_img, filtered_imgs, interpolation="nearest"
        )
        masker = NiftiMasker(
            mask_img=resampled_mask,
            dtype=np.float32,
            standardize=False,
            smoothing_fwhm=None,
        )
        X = masker.fit_transform(filtered_imgs)

        # run cross-validation
        classifier = LinearSVC(
            C=SVC_C, random_state=SVC_RANDOM_STATE, max_iter=SVC_MAX_ITER
        )
        cv_results = cross_validate(
            classifier,
            X,
            filtered_labels,
            groups=filtered_subjs,  # LO-Group-O cv
            cv=cv,
            n_jobs=n_jobs,
            return_estimator=False,
            scoring="accuracy",
        )

        cv_scores = cv_results["test_score"]
        max_acc = np.max(cv_scores)
        top_accuracies.append(max_acc)

        # save individual fold accuracies
        fold_accs_path = results_path / f"perm_{perm_idx:04d}_fold_accuracies.json"
        with open(fold_accs_path, "w") as f:
            json.dump(
                {
                    "fold_accuracies": cv_scores.tolist(),
                    "mean": float(np.mean(cv_scores)),
                    "max": float(max_acc),
                },
                f,
            )
        logger.debug(
            f"saved permutation {perm_idx+1} fold accuracies to {fold_accs_path}"
        )

        # save results every 10 permutations
        if (perm_idx + 1) % 10 == 0:
            logger.info(
                f"completed {perm_idx+1} permutations, mean top accuracy: {np.mean(top_accuracies):.4f}"
            )

    top_accuracies = np.array(top_accuracies)

    # save top accuracies list
    top_accuracies_path = results_path / "top_accuracies.npy"
    np.save(top_accuracies_path, top_accuracies)
    logger.info(f"saved top accuracies to {top_accuracies_path}")

    # create and save top accuracy histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(top_accuracies, bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Max Fold Accuracy")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of Top Accuracies ({n_permutations} permutations)")
    # add mean line
    ax.axvline(
        np.mean(top_accuracies),
        color="r",
        linestyle="--",
        label=f"Mean: {np.mean(top_accuracies):.4f}",
    )
    ax.legend()

    histogram_path = results_path / "top_accuracies_histogram.png"
    plt.savefig(histogram_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"saved histogram to {histogram_path}")

    # save summary
    summary = {
        "n_permutations": int(n_permutations),
        "labels_used": labels_to_use,
        "top_accuracies": {
            "mean": float(np.mean(top_accuracies)),
            "std": float(np.std(top_accuracies)),
            "min": float(np.min(top_accuracies)),
            "max": float(np.max(top_accuracies)),
            "median": float(np.median(top_accuracies)),
        },
        "timestamp": datetime.now().isoformat(),
    }

    summary_path = results_path / "permutation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"saved summary to {summary_path}")

    logger.info(
        f"permutation test completed. Mean top accuracy: {np.mean(top_accuracies):.4f} +/- {np.std(top_accuracies):.4f}"
    )

    return top_accuracies


def permutation_test_searchlight_within_participant(
    betas_dir: Path,
    labels_to_use: List[str],
    mask_path: Path,
    results_path: Path,
    subject_id: str,
    n_permutations: int = 100,
    radius: int = 8,
    n_jobs: int = 4,
    process_mask_path: Path | None = None,
    random_state: int | None = None,
    logger: logging.Logger | None = None,
):
    """run within-subject searchlight permutation test by shuffling labels.

    parameters:
    betas_dir : Path
        directory containing each subjects nifti folder.
    labels_to_use : List[str]
        labels/emotions to evaluate.
    mask_path : Path
        brain mask path used for resampling/masking.
    results_path : Path
        directory where permutation outputs are written.
    subject_id : str
        subject identifier to restrict the analysis to.
    n_permutations : int
        number of permutations to run.
    radius : int
        searchlight radius in mm.
    n_jobs : int
        number of parallel jobs.
    process_mask_path : Path | None
        optional process mask path to restrict searchlight.
    random_state : int | None
        random seed for reproducibility.
    logger : logging.Logger | None
        optional logger for progress messages.

    returns:
    list, list
        list of top accuracies and list of searchlight score images.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)

    # set random state for reproducibility
    rng = np.random.RandomState(random_state)

    results_path.mkdir(parents=True, exist_ok=True)

    # load all data and filter for subject
    logger.info(f"loading data for subject {subject_id}")
    all_data_df = load_all_participants(betas_dir)
    subject_data_df = all_data_df[all_data_df["subj"] == subject_id].copy()

    if len(subject_data_df) == 0:
        raise ValueError(f"no data found for subject {subject_id}")

    logger.info(f"loaded {len(subject_data_df)} beta entries for subject {subject_id}")

    if random_state is not None:
        logger.info(f"using random_state={random_state}")

    top_accuracies = []
    accuracy_maps = []

    logger.info(f"running {n_permutations} permutation iterations")

    for perm_idx in range(n_permutations):
        logger.info(f"permutation {perm_idx+1}/{n_permutations}")

        # Permute labels within subject data
        permuted_df = subject_data_df.copy()
        permuted_df["label"] = rng.permutation(permuted_df["label"].values)

        # Filter by emotions
        try:
            filtered_imgs, filtered_labels, filtered_subjs, filtered_runs = (
                filter_by_emotions(permuted_df, labels_to_use)
            )
        except ValueError as e:
            logger.warning(
                f"perm {perm_idx+1} produced invalid label distribution: {e}"
            )
            continue

        if len(np.unique(filtered_labels)) < 2:
            logger.warning(
                f"perm {perm_idx+1} has less than 2 unique labels after filtering"
            )
            continue

        # apply mask
        masked_imgs = apply_mask(filtered_imgs, mask_path)
        process_mask = get_process_mask(process_mask_path, masked_imgs)

        # run searchlight with run-based LOGO CV
        sl = SearchLight(
            scoring="accuracy",
            cv=LeaveOneGroupOut(),
            n_jobs=n_jobs,
            verbose=0,
            radius=radius,
            process_mask_img=process_mask,
        )

        sl.fit(masked_imgs, filtered_labels, groups=filtered_runs)
        scores_img = sl.scores_img_
        top_accuracies.append(np.max(sl.scores_))
        accuracy_maps.append(scores_img)

        # save individual permutation map
        perm_scores_path = results_path / f"perm_{perm_idx:04d}_scores.nii.gz"
        scores_img.to_filename(perm_scores_path)

    # create histogram from top accuracies
    if top_accuracies:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(top_accuracies, bins=20, alpha=0.7, edgecolor="black")
        ax.set_xlabel("Top Searchlight Accuracy")
        ax.set_ylabel("Frequency")
        ax.set_title(
            f"Permutation Test - Within Subject {subject_id} Searchlight (n={len(top_accuracies)})"
        )
        plt.tight_layout()
        hist_path = results_path / "permutation_histogram.png"
        plt.savefig(hist_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"saved histogram to {hist_path}")

        # Save summary
        summary = {
            "subject_id": subject_id,
            "n_permutations": len(top_accuracies),
            "mean_top_accuracy": float(np.mean(top_accuracies)),
            "std_top_accuracy": float(np.std(top_accuracies)),
            "min_top_accuracy": float(np.min(top_accuracies)),
            "max_top_accuracy": float(np.max(top_accuracies)),
            "labels_used": labels_to_use,
        }
        summary_path = results_path / "permutation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {summary_path}")

    return top_accuracies, accuracy_maps


def permutation_test_wholebrain_within_participant(
    betas_dir: Path,
    labels_to_use: List[str],
    mask_path: Path,
    results_path: Path,
    subject_id: str,
    n_permutations: int = 100,
    n_jobs: int = 1,
    random_state: int | None = None,
    logger: logging.Logger | None = None,
):
    """run within-subject whole-brain permutation test by shuffling labels.

    parameters:
    betas_dir : Path
        directory containing beta nifti files organized per subject.
    labels_to_use : List[str]
        labels/emotions to evaluate.
    mask_path : Path
        brain mask path used for resampling/masking.
    results_path : Path
        directory where permutation outputs are written.
    subject_id : str
        subject identifier to restrict the analysis to.
    n_permutations : int
        number of permutations to run.
    n_jobs : int
        number of parallel jobs for cross-validation.
    random_state : int | None
        random seed for reproducibility.
    logger : logging.Logger | None
        optional logger for progress messages.

    returns:
    list
        list of top fold accuracies for each permutation.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)

    # set random state for reproducibility
    rng = np.random.RandomState(random_state)

    results_path.mkdir(parents=True, exist_ok=True)

    # load all data and filter to subject
    logger.info(f"Loading data for subject {subject_id}")
    all_data_df = load_all_participants(betas_dir)
    subject_data_df = all_data_df[all_data_df["subj"] == subject_id].copy()

    if len(subject_data_df) == 0:
        raise ValueError(f"no data found for subject {subject_id}")

    logger.info(f"loaded {len(subject_data_df)} beta entries for subject {subject_id}")

    if random_state is not None:
        logger.info(f"using random_state={random_state} for reproducibility")

    top_accuracies = []

    logger.info(f"running {n_permutations} permutation iterations")

    for perm_idx in range(n_permutations):
        logger.info(f"permutation {perm_idx+1}/{n_permutations}")

        # permute labels within subject data
        permuted_df = subject_data_df.copy()
        permuted_df["label"] = rng.permutation(permuted_df["label"].values)

        # filter by emotions
        try:
            filtered_imgs, filtered_labels, filtered_subjs, filtered_runs = (
                filter_by_emotions(permuted_df, labels_to_use)
            )
        except ValueError as e:
            logger.warning(
                f"perm {perm_idx+1} produced invalid label distribution: {e}"
            )
            continue

        if len(np.unique(filtered_labels)) < 2:
            logger.warning(
                f"perm {perm_idx+1} has less than 2 unique labels after filtering"
            )
            continue

        # extract voxel features using NiftiMasker
        mask_img = load_img(mask_path)
        resampled_mask = resample_to_img(
            mask_img, filtered_imgs, interpolation="nearest"
        )
        masker = NiftiMasker(
            mask_img=resampled_mask,
            dtype=np.float32,
            standardize=False,
            smoothing_fwhm=None,
        )
        X = masker.fit_transform(filtered_imgs)

        # run wholebrain with run-based LOGO CV
        cv = LeaveOneGroupOut()
        svc = LinearSVC(max_iter=SVC_MAX_ITER, C=SVC_C, random_state=SVC_RANDOM_STATE)

        cv_results = cross_validate(
            svc,
            X,
            filtered_labels,
            groups=filtered_runs,
            cv=cv,
            scoring="accuracy",
            n_jobs=n_jobs,
            return_estimator=False,
        )

        cv_scores = cv_results["test_score"]
        top_accuracies.append(np.max(cv_scores))

        # save fold accuracies for this permutation
        fold_data = {
            "fold_accuracies": cv_scores.tolist(),
            "mean": float(np.mean(cv_scores)),
            "max": float(np.max(cv_scores)),
        }
        fold_path = results_path / f"perm_{perm_idx:04d}_fold_accuracies.json"
        with open(fold_path, "w") as f:
            json.dump(fold_data, f, indent=2)

    # create histogram
    if top_accuracies:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(top_accuracies, bins=20, alpha=0.7, edgecolor="black")
        ax.set_xlabel("Max CV Accuracy")
        ax.set_ylabel("Frequency")
        ax.set_title(
            f"Permutation Test - Within Subject {subject_id} Wholebrain (n={len(top_accuracies)})"
        )
        plt.tight_layout()
        hist_path = results_path / "permutation_histogram.png"
        plt.savefig(hist_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"saved histogram to {hist_path}")

        # save summary
        summary = {
            "subject_id": subject_id,
            "n_permutations": len(top_accuracies),
            "mean_top_accuracy": float(np.mean(top_accuracies)),
            "std_top_accuracy": float(np.std(top_accuracies)),
            "min_top_accuracy": float(np.min(top_accuracies)),
            "max_top_accuracy": float(np.max(top_accuracies)),
            "labels_used": labels_to_use,
        }
        summary_path = results_path / "permutation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"saved summary to {summary_path}")

    return top_accuracies
