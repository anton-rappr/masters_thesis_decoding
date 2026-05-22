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
from data.masking import apply_mask, get_process_mask, create_single_slice_mask
from data.data_saving import save_decoding_results, save_wholebrain_results

TIME_ID = str(datetime.now().strftime(format="%d-%m_%H-%M"))

# SVC hyperparameters used for each training in the functions below as well
# as the permutation testing
SVC_MAX_ITER = 1000
SVC_C = 1
SVC_RANDOM_STATE = 42


def train_searchlight(
    betas_dir: Path,
    labels_to_use: List[str],
    results_path: Path,
    mask_path: Path,
    radius: int,
    n_jobs: int = 4,
    process_mask_path: Path | None = None,
    one_slice_only: bool = False,
    logger: logging.Logger | None = None,
):
    """train a searchlight decoder using linear svm and leave-one-group-out cv.

    parameters:
    betas_dir : Path
        directory containing each subjects nifti folder.
    labels_to_use : List[str]
        labels to decode.
    results_path : Path
        directory to write results and metadata.
    mask_path : Path
        brain mask path used to align images.
    radius : int
        searchlight radius in mm.
    n_jobs : int
        number of parallel jobs.
    process_mask_path : Path | None
        optional path to a process mask to restrict analysis.
    one_slice_only : bool
        if true, run searchlight on a single slice for quick tests.
    logger : logging.Logger | None
        optional logger instance.

    returns:
    SearchLight, dict
        trained SearchLight object and metadata about the training run.
    """

    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)

    # get file, label, block, run and subj for each nii file in the BETAS dir and its subdirs
    all_data_df = load_all_participants(betas_dir)
    logger.info(f"loaded {len(all_data_df)} total beta entries from {betas_dir}")

    # filter it by the provided labels
    filtered_imgs, filtered_labels, filtered_subjs, filtered_runs = filter_by_emotions(
        all_data_df, labels_to_use
    )
    logger.info(
        f"selected {len(filtered_labels)} samples from {len(np.unique(filtered_subjs))} subjects for labels {labels_to_use}"
    )

    # mask images
    masked_imgs = apply_mask(filtered_imgs, mask_path)

    # resample process mask if supplied
    if one_slice_only:
        logger.info(
            "one_slice_only is set to true: running searchlight analysis only on one slice of the data"
        )
        process_mask = create_single_slice_mask(
            mask_path, 30, save_path=results_path / "processing_mask.nii"
        )
    else:
        process_mask = get_process_mask(process_mask_path, masked_imgs)

    cv = LeaveOneGroupOut()
    logger.info(
        f"initializing Searchlight object with n_jobs={n_jobs} and leave-one-group-out CV"
    )
    searchlight_object = SearchLight(
        masked_imgs,
        process_mask_img=process_mask,  # is none if no path is supplied
        radius=radius,
        n_jobs=n_jobs,
        verbose=1,
        cv=cv,
    )
    logger.info(
        f"running searchlight analysis with radius {searchlight_object.radius} and betas_dir={betas_dir}"
    )

    for i, (train_idx, test_idx) in enumerate(
        cv.split(np.zeros(len(filtered_labels)), filtered_labels, groups=filtered_subjs)
    ):
        logger.info(
            f"fold {i+1}: {len(train_idx)} train samples, {len(test_idx)} test samples"
        )
        logger.debug(f"test_idx {i+1}: {test_idx}")

    start_time = time.time()
    searchlight_object.fit(filtered_imgs, filtered_labels, groups=filtered_subjs)
    training_time_seconds = time.time() - start_time
    logger.info(
        f"searchlight training completed in {training_time_seconds:.2f} seconds"
    )

    save_decoding_results(results_path, searchlight_object, TIME_ID)

    train_metadata = {
        "n_total_samples": int(len(all_data_df)),
        "n_selected_samples": int(len(filtered_labels)),
        "n_subjects": int(len(np.unique(filtered_subjs))),
        "labels_to_use": labels_to_use,
        "unique_labels": [str(label) for label in np.unique(filtered_labels)],
        "radius": radius,
        "n_jobs": n_jobs,
        "one_slice_only": one_slice_only,
        "results_path": str(results_path),
        "finished_at": datetime.now().isoformat(),
        "training_time_seconds": training_time_seconds,
    }

    metadata_file = results_path / "train_metadata.json"
    try:
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(train_metadata, f, indent=2)
        logger.info(f"train metadata saved to {metadata_file}")
    except Exception as e:
        logger.warning(f"failed to save train metadata to {metadata_file}: {e}")

    return searchlight_object, train_metadata


def train_wholebrain(
    betas_dir: Path,
    labels_to_use: List[str],
    results_path: Path,
    mask_path: Path,
    n_jobs: int = 1,
    experiment_name: str | None = None,
):
    """train a whole-brain linear svm decoder using leave-one-group-out cv.

    parameters:
    betas_dir : Path
        directory containing each subjects nifti folder.
    labels_to_use : List[str]
        labels or conditions to decode.
    results_path : Path
        directory to write results and metadata.
    mask_path : Path
        brain mask path used to extract voxel features.
    n_jobs : int
        number of parallel jobs for cross-validation.
    experiment_name : str | None
        optional prefix for result filenames.

    returns:
    np.ndarray, nibabel.Nifti1Image
        cross-validation scores and averaged coefficient image.
    """

    all_data_df = load_all_participants(betas_dir)
    filtered_imgs, filtered_labels, filtered_subjs, _ = filter_by_emotions(
        all_data_df, labels_to_use
    )
    del all_data_df

    print(
        "wholebrain filter labels unique/counts:",
        np.unique(filtered_labels, return_counts=True),
    )
    print("wholebrain subjects unique:", np.unique(filtered_subjs))
    print(
        "wholebrain labels shape:",
        filtered_labels.shape,
        "subjects shape:",
        filtered_subjs.shape,
    )

    mask_img = load_img(mask_path)
    resampled_mask = resample_to_img(mask_img, filtered_imgs, interpolation="nearest")

    masker = NiftiMasker(
        mask_img=resampled_mask,
        dtype=np.float32,
        standardize=False,
        smoothing_fwhm=None,
    )

    print(f"fitting masker to {filtered_imgs.shape[3]} images with mask {mask_path}")

    X = masker.fit_transform(filtered_imgs)
    del filtered_imgs

    cv = LeaveOneGroupOut()
    classifier = LinearSVC(
        C=SVC_C, random_state=SVC_RANDOM_STATE, max_iter=SVC_MAX_ITER
    )

    print(
        f"initializing whole-brain decoder with n_jobs={n_jobs} and leave one group out"
    )

    # collect held-out subjects in fold order for later summary / file naming
    fold_heldout_subjects = []
    for _, test_idx in cv.split(
        np.zeros(len(filtered_labels)), filtered_labels, groups=filtered_subjs
    ):
        held_out = np.unique(filtered_subjs[test_idx])
        if len(held_out) != 1:
            raise ValueError(
                f"expected one held-out subject per test fold, got {held_out}"
            )
        fold_heldout_subjects.append(str(held_out[0]))

    # run cross-validation with parallelization and return trained estimators
    start_time = time.time()
    cv_results = cross_validate(
        classifier,
        X,
        filtered_labels,
        groups=filtered_subjs,
        cv=cv,
        n_jobs=n_jobs,
        return_estimator=True,
        scoring="accuracy",
    )
    training_time_seconds = time.time() - start_time

    cv_scores = cv_results["test_score"]
    fold_classifiers = cv_results["estimator"]

    print(f"whole-brain CV scores: {cv_scores}")
    print(f"whole-brain mean accuracy: {np.mean(cv_scores):.4f}")
    print(f"held-out subjects per fold: {fold_heldout_subjects}")

    # convert coefficients from each fold to image space and collect
    fold_coef_maps = []
    for fold_classifier in fold_classifiers:
        fold_coef_img = masker.inverse_transform(fold_classifier.coef_.ravel())
        fold_coef_maps.append(fold_coef_img)

    # average coefficients across all CV folds
    individual_coefs = np.array([fc.coef_.ravel() for fc in fold_classifiers])
    avg_coef = np.mean(individual_coefs, axis=0)
    coef_img = masker.inverse_transform(avg_coef)

    result_prefix = experiment_name or "wholebrain"
    print(f"whole-brain output prefix: {result_prefix}")

    save_wholebrain_results(
        results_path=results_path,
        coef_img=coef_img,
        cv_scores=cv_scores,
        labels_used=list(labels_to_use),
        result_prefix=result_prefix,
        fold_coef_maps=fold_coef_maps,
        fold_heldout_subjects=fold_heldout_subjects,
        svc_max_iter=SVC_MAX_ITER,
        svc_C=SVC_C,
        svc_random_state=SVC_RANDOM_STATE,
        training_time_seconds=training_time_seconds,
    )

    return cv_scores, coef_img


def train_searchlight_within_participant(
    betas_dir: Path,
    labels_to_use: List[str],
    results_path: Path,
    mask_path: Path,
    subject_id: str,
    radius: int,
    n_jobs: int = 4,
    process_mask_path: Path | None = None,
    one_slice_only: bool = False,
    logger: logging.Logger | None = None,
):
    """train a within-subject searchlight decoder using leave-one-run-out cv.

    parameters:
    betas_dir : Path
        directory containing each subjects nifti folder.
    labels_to_use : List[str]
        labels or conditions to decode.
    results_path : Path
        directory to write results and metadata.
    mask_path : Path
        brain mask path used to align images.
    subject_id : str
        subject identifier to analyze.
    radius : int
        searchlight radius in mm.
    n_jobs : int
        number of parallel jobs.
    process_mask_path : Path | None
        optional path to a process mask to restrict analysis.
    one_slice_only : bool
        if true, run searchlight on a single slice for quick tests.
    logger : logging.Logger | None
        optional logger instance.

    returns:
    SearchLight, dict
        trained SearchLight object and metadata about the training run.
    """

    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)

    # get file, label, block, run and subj for each nii file in the BETAS dir and its subdirs
    all_data_df = load_all_participants(betas_dir)
    # filter to only the specified subject
    all_data_df = all_data_df[all_data_df["subj"] == subject_id]
    logger.info(
        f"loaded {len(all_data_df)} total beta entries for subject {subject_id} from {betas_dir}"
    )

    # filter it by the provided labels
    filtered_imgs, filtered_labels, _, filtered_runs = filter_by_emotions(
        all_data_df, labels_to_use
    )
    logger.info(
        f"Selected {len(filtered_labels)} samples from subject {subject_id} for labels {labels_to_use}"
    )

    # Use runs as groups for leave-one-run-out within this participant
    groups = filtered_runs

    # mask images
    masked_imgs = apply_mask(filtered_imgs, mask_path)

    # resample process mask if supplied
    if one_slice_only:
        logger.info(
            "one_slice_only is set to true: running searchlight analysis only on one slice of the data"
        )
        process_mask = create_single_slice_mask(
            mask_path, 30, save_path=results_path / "processing_mask.nii"
        )
    else:
        process_mask = get_process_mask(process_mask_path, masked_imgs)

    cv = LeaveOneGroupOut()
    logger.info(
        f"initializing Searchlight object with n_jobs={n_jobs} and leave-one-group-out CV (within-participant)"
    )
    searchlight_object = SearchLight(
        masked_imgs,
        process_mask_img=process_mask,  # is none if no path is supplied
        radius=radius,
        n_jobs=n_jobs,
        verbose=1,
        cv=cv,
    )
    logger.info(
        f"running searchlight analysis with radius {searchlight_object.radius} and betas_dir={betas_dir}"
    )

    for i, (train_idx, test_idx) in enumerate(
        cv.split(np.zeros(len(filtered_labels)), filtered_labels, groups=groups)
    ):
        logger.info(
            f"Fold {i+1}: {len(train_idx)} train samples, {len(test_idx)} test samples"
        )
        logger.debug(f"test_idx {i+1}: {test_idx}")

    start_time = time.time()
    searchlight_object.fit(filtered_imgs, filtered_labels, groups=groups)
    training_time_seconds = time.time() - start_time
    logger.info(
        f"searchlight training completed in {training_time_seconds:.2f} seconds"
    )

    save_decoding_results(results_path, searchlight_object, TIME_ID)

    train_metadata = {
        "subject_id": subject_id,
        "n_total_samples": int(len(all_data_df)),
        "n_selected_samples": int(len(filtered_labels)),
        "labels_to_use": labels_to_use,
        "unique_labels": [str(label) for label in np.unique(filtered_labels)],
        "radius": radius,
        "n_jobs": n_jobs,
        "one_slice_only": one_slice_only,
        "results_path": str(results_path),
        "finished_at": datetime.now().isoformat(),
        "training_time_seconds": training_time_seconds,
        "cv_type": "within_participant_leave_one_run_out",
    }

    metadata_file = results_path / "train_metadata.json"
    try:
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(train_metadata, f, indent=2)
        logger.info(f"train metadata saved to {metadata_file}")
    except Exception as e:
        logger.warning(f"failed to save train metadata to {metadata_file}: {e}")

    return searchlight_object, train_metadata


def train_wholebrain_within_participant(
    betas_dir: Path,
    labels_to_use: List[str],
    results_path: Path,
    mask_path: Path,
    subject_id: str,
    n_jobs: int = 1,
    experiment_name: str | None = None,
):
    """train a within-subject whole-brain linear svm decoder using run-based cv.

    parameters:
    betas_dir : Path
        directory containing each subjects nifti folder.
    labels_to_use : List[str]
        labels or conditions to decode.
    results_path : Path
        directory to write results and metadata.
    mask_path : Path
        brain mask path used to extract voxel features.
    subject_id : str
        subject identifier to analyze.
    n_jobs : int
        number of parallel jobs for cross-validation.
    experiment_name : str | None
        optional prefix for result filenames.

    returns:
    np.ndarray, nibabel.Nifti1Image
        cross-validation scores and averaged coefficient image.
    """

    all_data_df = load_all_participants(betas_dir)
    # filter to only the specified subject
    all_data_df = all_data_df[all_data_df["subj"] == subject_id]
    filtered_imgs, filtered_labels, _, filtered_runs = filter_by_emotions(
        all_data_df, labels_to_use
    )
    del all_data_df

    # use runs as groups for leave-one-run-out within this participant
    groups = filtered_runs

    print(
        "wholebrain within-participant filter labels unique/counts:",
        np.unique(filtered_labels, return_counts=True),
    )
    print("wholebrain subject:", subject_id)
    print("wholebrain runs/groups unique:", np.unique(filtered_runs))
    print(
        "wholebrain labels shape:", filtered_labels.shape, "groups shape:", groups.shape
    )

    mask_img = load_img(mask_path)
    resampled_mask = resample_to_img(mask_img, filtered_imgs, interpolation="nearest")

    masker = NiftiMasker(
        mask_img=resampled_mask,
        dtype=np.float32,
        standardize=False,
        smoothing_fwhm=None,
    )

    print(f"fitting masker to {filtered_imgs.shape[3]} images with mask {mask_path}")

    X = masker.fit_transform(filtered_imgs)
    del filtered_imgs

    cv = LeaveOneGroupOut()

    classifier = LinearSVC(
        C=SVC_C, random_state=SVC_RANDOM_STATE, max_iter=SVC_MAX_ITER
    )

    print(
        f"initializing whole-brain decoder with n_jobs={n_jobs} and leave one group out (within-participant)"
    )

    # collect held-out runs in fold order for later summary / file naming
    fold_heldout_runs = []
    for _, test_idx in cv.split(
        np.zeros(len(filtered_labels)), filtered_labels, groups=groups
    ):
        held_out = np.unique(groups[test_idx])
        if len(held_out) != 1:
            raise ValueError(f"expected one held-out run per test fold, got {held_out}")
        fold_heldout_runs.append(str(held_out[0]))

    # run cross-validation with parallelization and return trained estimators
    start_time = time.time()
    cv_results = cross_validate(
        classifier,
        X,
        filtered_labels,
        groups=groups,
        cv=cv,
        n_jobs=n_jobs,
        return_estimator=True,
        scoring="accuracy",
    )
    training_time_seconds = time.time() - start_time

    cv_scores = cv_results["test_score"]
    fold_classifiers = cv_results["estimator"]

    print(f"whole-brain CV scores: {cv_scores}")
    print(f"whole-brain mean accuracy: {np.mean(cv_scores):.4f}")
    print(f"held-out runs per fold: {fold_heldout_runs}")

    # convert coefficients from each fold to image space and collect
    fold_coef_maps = []
    for fold_classifier in fold_classifiers:
        fold_coef_img = masker.inverse_transform(fold_classifier.coef_.ravel())
        fold_coef_maps.append(fold_coef_img)

    # average coefficients across all CV folds
    individual_coefs = np.array([fc.coef_.ravel() for fc in fold_classifiers])
    avg_coef = np.mean(individual_coefs, axis=0)
    coef_img = masker.inverse_transform(avg_coef)

    result_prefix = experiment_name or "wholebrain_within_participant"
    print(f"whole-brain output prefix: {result_prefix}")

    save_wholebrain_results(
        results_path=results_path,
        coef_img=coef_img,
        cv_scores=cv_scores,
        labels_used=list(labels_to_use),
        result_prefix=result_prefix,
        fold_coef_maps=fold_coef_maps,
        fold_heldout_subjects=fold_heldout_runs,  # using runs instead of subjects
        svc_max_iter=SVC_MAX_ITER,
        svc_C=SVC_C,
        svc_random_state=SVC_RANDOM_STATE,
        training_time_seconds=training_time_seconds,
    )

    return cv_scores, coef_img
