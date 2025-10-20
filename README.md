# Event Log Extraction from Text

This repository contains the code and resources used in our research paper:

**_Extracting Events from Nursing Notes: A MIMIC-III Case Study_**  
by **Allmin Susaiyah** and **Natalia Sidorova**

📄 **Download the paper:** [https://www.genai4pm2025.info/179.pdf](https://www.genai4pm2025.info/179.pdf)

---

In this project, we investigate the selective use of language models to extract structured event logs from unstructured clinical text — specifically, nursing notes from the MIMIC-III dataset.

---

## Setup Instructions

1. **Clone the repository**
    ```bash
    git clone https://github.com/allmin/event_log_from_text.git
    cd event_log_from_text
    ```

2. **Set up Ollama**
    - Follow the instructions at [https://github.com/ollama/ollama](https://github.com/ollama/ollama) to install and configure Ollama on your system.

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download spaCy language model**
    ```bash
    python -m spacy download en_core_web_lg
    ```

 5. **Prepare MIMIC CSV Files**

- Extract all MIMIC CSV files into a folder.
- In the main project directory, create a `.env` file and specify the path to the extracted folder as follows:

  ```dotenv
  MIMICPATH=/path/to/mimic/csvs

6. **Run scripts and notebooks**
    - Execute all scripts and notebooks in the `scripts` directory in order (e.g., `01_*.py`, `02_*.ipynb`, ...).
    ```bash
    cd scripts
    <open notebook/scipt and run>
    ```

7. **Locate event logs**
    - Extracted event logs will be available in the `exports` folder.

## Notes

- This project was developed using Python 3.12.1.
- Ensure all dependencies are installed and environment variables are set before running the scripts.
- For questions or issues, please refer to the repository's issue tracker. Event Log Extraction from Text
