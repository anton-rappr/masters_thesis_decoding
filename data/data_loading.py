from nilearn.image import index_img, new_img_like
import pandas as pd
from pathlib import Path
import numpy as np
import re
from typing import List


def load_label_file(subj_dir: Path, filter_mode: str = "listen") -> pd.DataFrame:
    """load labels from a subject's beta_name_mapping.csv and return a dataframe.
    the label file is expected to contain a regressor_name column that contains entries of the form 'Sn(1) listen_disgust_b1_R1*bf(1)', and a original_beta column that contains the file name.

    parameters:
    subj_dir : Path
        path to the subject folder containing `beta_name_mapping.csv`.
    filter_mode : str
        mode string to filter regressors (for example 'listen').

    returns:
    pd.DataFrame
        dataframe with columns ["file", "label", "block", "run", "subj"]
    """
    rows = []
    skipped = 0
    label_file = subj_dir / "beta_name_mapping.csv"

    if not label_file.exists():
        raise FileNotFoundError(f"file path {label_file} does not exist.")

    label_df = pd.read_csv(label_file)
    for _, row in label_df.iterrows():
        regressor_name = str(row.get("regressor_name", ""))
        if not regressor_name or regressor_name.lower() == "nan":
            skipped += 1
            continue

        regressors = re.split(
            r"[ _]+", regressor_name.strip()
        )  # split on either space or underscore
        if len(regressors) < 3:
            skipped += 1
            continue

        file = str(row.get("original_beta", ""))
        mode = regressors[1]
        emot = regressors[2]
        block = regressors[3]
        run = str(row.get("readable_name", "")).split("_")[0]

        if mode == filter_mode:
            rows.append(
                {
                    "file": subj_dir / file,
                    "label": f"#{mode}_{emot}",
                    "block": block,
                    "run": run,
                    "subj": subj_dir.name,
                }
            )

    df = pd.DataFrame(rows, columns=["file", "label", "block", "run", "subj"])
    if skipped > 0:
        print(
            f"skipped {skipped} rows in {label_file} because regressor_name could not be parsed."
        )
    print(f"loaded {len(df)} rows from {label_file} for mode={filter_mode}")
    return df


def load_all_participants(betas_dir: Path) -> pd.DataFrame:
    """load label data for all participant subdirectories in `betas_dir`.

    parameters:
    betas_dir : Path
        directory containing per-subject folders with beta mappings.

    returns:
    pd.DataFrame
        concatenated dataframe with columns ["file", "label", "block", "run", "subj"]
    """
    all_data = []
    for subj_dir in betas_dir.iterdir():
        if subj_dir.is_dir():
            subj_data = load_label_file(subj_dir)
            all_data.append(subj_data)
    all_data_df = pd.concat(all_data, ignore_index=True)
    print(f"Loaded all data: {all_data_df}")  # Debug
    return all_data_df


def filter_by_emotions(all_data_df: pd.DataFrame, labels_to_use: List[str]):
    """filter dataframe by provided emotion labels and return image and label arrays.

    parameters:
    all_data_df : pd.DataFrame
        dataframe with columns 'file', 'label', 'subj', 'run'.
    labels_to_use : List[str]
        list of labels to keep.

    returns:
    nibabel.Nifti1Image, np.ndarray, np.ndarray, np.ndarray
        filtered images, condition labels, subject ids, and run ids respectively.
    """
    # create binary condition mask
    cond_mask = all_data_df["label"].isin(labels_to_use)

    # condition mask to file df
    filtered_imgs = index_img(all_data_df["file"], cond_mask)

    # clean data of any nans
    clean_data = np.nan_to_num(filtered_imgs.get_fdata())
    filtered_imgs = new_img_like(filtered_imgs, clean_data)

    print("shape of loaded filtered imgs ", filtered_imgs.shape)  # Debug

    # filter labels to only get the labels that match the condition mask (to match filtered imgs)
    conditions = all_data_df["label"][cond_mask].to_numpy()
    filtered_subjs = all_data_df["subj"][cond_mask].to_numpy()
    filtered_runs = all_data_df["run"][cond_mask].to_numpy()

    if filtered_imgs.shape[-1] != len(conditions):
        raise ValueError(
            "label count does not match number of filtered images: "
            f"{len(conditions)} labels vs {filtered_imgs.shape[-1]} images."
        )
    if len(conditions) != len(filtered_subjs):
        raise ValueError(
            "number of labels does not match number of subjects for the filtered data."
        )

    print(f"filtered conditions: {conditions}")  # Debug
    print(f"subject nr for each img in filtered img: {filtered_subjs}")  # Debug
    print(f"runs for each img in filtered img: {filtered_runs}")  # Debug
    print(f"got unique subjects: {np.unique(filtered_subjs)}")  # Debug

    return filtered_imgs, conditions, filtered_subjs, filtered_runs
