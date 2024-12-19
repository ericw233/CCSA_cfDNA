# **CCSA-cfDNA**  
A Python implementation of CCSA ("Unified Deep Supervised Domain Adaptation and Generalization" ICCV 2017) for correcting batch effects in cfDNA genomic features used for multi-cancer early detection (MCED)

---

## **Table of Contents**
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

---

## **Features**
- Loads and preprocesses data for training and validation.
- Fits and validates a CCSA model.
- Conducts k-fold cross validations on the training frame.
- Outputs results in .csv format.

---

## **Installation**
### Prerequisites
- Python 3.8 or higher
- Git installed
- Conda installed

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/ericw233/CCSA_cfDNA.git
   cd CCSA_cfDNA

2. Create and activate a conda environment with dependencies
    ```bash
    conda env create -n CCSA_cfDNA_env -c conda-forge -f CCSA_cfDNA_env.yml
    conda activate CCSA_cfDNA_env

## **Usage**
1. Specify the directories and parameters in CCSA_run.sh

2. Run the scripts to process data and fit models
    ```bash
    nohup ./CCSA_run.sh >log_test.txt 2>&1 &

## **License**
This project is licensed under the MIT License.

## **Contact**
Created by Eric Wu
Email: nanoeric2@gmail.com

