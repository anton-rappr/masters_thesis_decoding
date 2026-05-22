from pathlib import Path
from nilearn.decoding import Decoder, SearchLight
from datetime import datetime
from nilearn.plotting import view_img, plot_img
from nilearn.image import (
    mean_img,
)
import json
import joblib
import numpy as np


def save_decoder(output_dir: Path, decoder: Decoder, label: str = "") -> Path:
    """save decoder coefficient image for a given label to `output_dir`.

    parameters:
    output_dir : Path
        directory to write the decoder weight image.
    decoder : Decoder
        nilearn Decoder object with fitted coefficients.
    label : str
        optional class label to select coefficients for.

    returns:
    Path
        path to the written nifti weight image.
    """

    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"output will be saved to: {output_dir}")

    if not label:
        label = decoder.classes_[0]  # default label to first decoder class

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_file_path = (
        output_dir / f"{label}_{decoder.estimator}_weights_{timestamp}.nii.gz"
    )

    decoder.coef_img_[label].to_filename(output_file_path)

    print(f"saved file {output_file_path}")
    return output_file_path


def save_decoding_results(
    results_path: Path,
    searchlight_object: SearchLight,
    time_id=str(datetime.now().strftime(format="%d-%m_%H-%M")),
):
    """save searchlight results: plot, nifti map, and serialized object.

    parameters:
    results_path : Path
        directory to save figures and outputs.
    searchlight_object : SearchLight
        trained nilearn SearchLight object.
    time_id : str
        identifier appended to output filenames.

    returns:
    None
    """

    scores_img = searchlight_object.scores_img_
    fig = plot_img(
        scores_img,
        title="Searchlight scores image",
        display_mode="tiled",
        # threshold=0.5,
        # vmin=0.5,
        # vmax=1.0,2,
        cmap="inferno",
        colorbar=True,
    )

    # save results plot cuts png
    results_path.mkdir(exist_ok=True)
    save_path = results_path / "acc_r{}_{}.png".format(
        searchlight_object.radius, time_id
    )
    print(f"saving accuracytrain_searchlight map figure to {save_path}")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")  #

    # save results accuracy whole brain nii image
    results_path.mkdir(exist_ok=True)
    save_path = results_path / "acc_r{}_{}.nii.gz".format(
        searchlight_object.radius, time_id
    )
    print(f"saving searchlight accuracy map in {save_path}")
    scores_img.to_filename(save_path)

    # save the searchlight object
    joblib.dump(
        searchlight_object,
        results_path
        / "sl_dec_r{}_t{}.pkl".format(
            searchlight_object.radius,
            str(datetime.now().strftime(format="%d-%m_%H-%M-%S")),
        ),
    )

    view_img(
        scores_img,
        title="Searchlight scores image",
        # threshold=0.5,
        # vmin=0.5,
        # vmax=1.0,
        colorbar=True,
        symmetric_cmap=False,
    ).open_in_browser()


def save_wholebrain_results(
    results_path: Path,
    coef_img,
    cv_scores: np.ndarray,
    labels_used: list[str],
    result_prefix: str = "wholebrain",
    time_id: str = str(datetime.now().strftime(format="%d-%m_%H-%M")),
    fold_coef_maps: list | None = None,
    fold_heldout_subjects: list | None = None,
    svc_max_iter: int | None = None,
    svc_C: float | None = None,
    svc_random_state: int | None = None,
    training_time_seconds: float | None = None,
):
    """save whole-brain results including coef maps, cv scores, and summary json.

    parameters:
    results_path : Path
        directory to save outputs.
    coef_img : nibabel.Nifti1Image
        averaged coefficient image to save.
    cv_scores : np.ndarray
        cross-validation accuracy scores per fold.
    labels_used : list[str]
        labels that were used for training.
    result_prefix : str
        prefix used for output filenames.
    time_id : str
        timestamp identifier appended to filenames.
    fold_coef_maps : list | None
        optional list of per-fold coefficient images.
    fold_heldout_subjects : list | None
        optional mapping of fold to heldout subject/run.
    svc_max_iter : int | None
    svc_C : float | None
    svc_random_state : int | None
    training_time_seconds : float | None

    returns:
    None
    """

    results_path.mkdir(exist_ok=True, parents=True)

    cv_scores_path = results_path / f"{result_prefix}_cv_scores_{time_id}.npy"
    np.save(cv_scores_path, cv_scores)

    # save averaged coefficient map
    coef_img_path = results_path / f"{result_prefix}_coef_map_averaged_{time_id}.nii.gz"
    coef_img.to_filename(coef_img_path)

    # save individual fold coefficient maps if provided
    if fold_coef_maps is not None:
        for i, fold_coef_img in enumerate(fold_coef_maps):
            subject_tag = ""
            if fold_heldout_subjects is not None and i < len(fold_heldout_subjects):
                subject_tag = f"_{str(fold_heldout_subjects[i]).replace(' ', '_')}"
            fold_coef_path = (
                results_path
                / f"{result_prefix}_coef_map_fold_{i:02d}{subject_tag}_{time_id}.nii.gz"
            )
            fold_coef_img.to_filename(fold_coef_path)
        print(f"saved {len(fold_coef_maps)} individual fold coefficient maps")

    fold_subjects = [str(s) for s in (fold_heldout_subjects or [])]
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "result_prefix": result_prefix,
        "labels_used": labels_used,
        "n_folds": int(len(cv_scores)),
        "SVC max_iter": svc_max_iter,
        "SVC C": svc_C,
        "SVC random_state": svc_random_state,
        "training_time_seconds": training_time_seconds,
        "cv_accuracy": {
            "mean": float(np.mean(cv_scores)),
            "std": float(np.std(cv_scores)),
            "min": float(np.min(cv_scores)),
            "max": float(np.max(cv_scores)),
            "median": float(np.median(cv_scores)),
        },
        "folds": [
            {
                "fold": i + 1,
                "heldout_subject": fold_subjects[i] if i < len(fold_subjects) else None,
                "accuracy": float(cv_scores[i]),
            }
            for i in range(len(cv_scores))
        ],
    }

    summary_path = results_path / f"{result_prefix}_summary_{time_id}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"saved whole-brain CV scores to {cv_scores_path}")
    print(f"saved whole-brain summary to {summary_path}")
