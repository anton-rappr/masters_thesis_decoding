# Cross-Participant Decoding of Emotions Conveyed by Nonverbal Emotional Vocalizations - Code 
This repository contains the code used for the analyses performed in the masters thesis with the title "Cross-Participant Decoding of Emotions Conveyed by Nonverbal Emotional Vocalizations" by Anton Rapprich. 
It contains functions for both within- and across-participant searchlight analysis as well as whole-brain MVPA using linearSVC's.
Furthermore, functions for permutation testing are included in the repository.

Note: The code used is specialized for the dataset described in the thesis. Unless you obtain access to the dataset, which is currently (last checked: 22nd of May 2026) not publicly available, this repository serves more as a way to understand and reproduce the analysis performed and might need to be changed in order to be used in combination with other datasets.


## Dependency installation
The dependencies used in this project can be found in the `pyproject.toml` file. The recommended way of installing the dependencies is through the python package manager `uv`, for which the installation is described below.

### installation with uv 
First, install the package manager `uv` as described on their website, see here: https://docs.astral.sh/uv/getting-started/installation/

Next, open a terminal and enter the repository and then enter the command
```uv sync```
This will create a virtual environment with all the necessary dependencies.

Now, the `main.py` file can be executed using the command
```uv run main.py```