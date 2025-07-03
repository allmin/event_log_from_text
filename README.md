# Event Log Extraction from Text

This project focuses on extracting event logs from textual data, specifically using MIMIC-III CSV files.

## Setup Instructions

1. **Clone the repository**
    ```bash
    git clone <repository-url>
    cd event_log_from_text
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download spaCy language model**
    ```bash
    python -m spacy download en_core_web_lg
    ```

4. **Prepare MIMIC CSV files**
    - Extract all MIMIC CSV files into a folder.
    - Set the environment variable `MIMICSPATH` to the path of this folder:
      ```bash
      export MIMICSPATH=/path/to/mimic/csvs
      ```

5. **Run scripts and notebooks**
    - Execute all scripts and notebooks in the `scripts` directory in order (e.g., `01_*.py`, `02_*.ipynb`, ...).

6. **Locate event logs**
    - Extracted event logs will be available in the `exports` folder.

## Notes

- This project was developed using Python 3.12.1.
- Ensure all dependencies are installed and environment variables are set before running the scripts.
- For questions or issues, please refer to the repository's issue tracker.
