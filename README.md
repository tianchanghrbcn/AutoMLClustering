# AutoMLClustering

This project implements an automated machine learning pipeline for clustering tasks, supporting various data preprocessing, error correction, clustering methods, and result analysis.

## Prerequisites

- Python 3.8 or above
- Linux-based system (tested on Ubuntu)
- Required packages (installed via `config.sh`)

## Installation

1. **Clone the repository:**

   ```bash
   git clone git@github.com:tianchanghrbcn/AutoMLClustering.git
   cd /home/changtian/Cleaning-Clustering
   ```

2. **Run the configuration script** to set up the virtual environment and install dependencies (about 15min on a 8vCPUs computer):

   ```bash
   bash init_config.sh
   ```
   *If the first attempt fails (e.g. interrupted download or partially‑set environment), simply re‑run the setup with `bash config.sh`.*

3. **Activate the virtual environment:**

   ```bash
   conda activate torch110
   ```

### Step 1 — Data Preprocessing

```bash
cd src/pipeline/train
python pre-processing.py
```


### Step 2 — Start the Training Offline Comparative Experiment


```bash
nohup python train_pipeline.py > output_training.log 2>&1 &
```

* `PYTHONPATH` is assumed to be permanently set.
* Training output appears in `output_training.log`.


### Step 3 — Analyze Training Results

After `train_pipeline.py` finishes, execute the analysis suite in **the same directory**:

```bash
python GroundTruth.py
python comparison.py
python ../utils/analyze_cleaning.py
python ../utils/analyze_cluster.py
python ../utils/merge_form.py
```

Result files and plots are saved to
`/home/changtian/Cleaning-Clustering/results/analysis_results`.


### Step 4 — Create and Activate the Classifier Environment

A dedicated environment isolates heavy ML libraries from the rest of the project.

```bash
conda create -y -n train39 python=3.9
conda activate train39
pip install numpy pandas scikit-learn lightgbm optuna joblib openpyxl
```

### Step 5 — Start the Classifier

Stay in `src/pipeline/train` and keep the `train39` environment active:

```bash
nohup python classifier.py > output_classifier.log 2>&1 &
```


### Step 6 — Run the Search Script

Still inside the `train39` environment and the same directory:

```bash
nohup python search.py > output_search.log 2>&1 &
```

### Step 7 — Run the Test Pipeline

```bash
cd ../test          # now in src/pipeline/test
nohup python test_pipeline.py > output_testing.log 2>&1 &
```


### Step 8 — Compute Loss and Accuracy

```bash
python compute_loss_and_acc.py
```

This script reads the test outputs and prints final loss/accuracy metrics.


### Additional Notes

1. **Check Running Processes:**
   To ensure that the scripts are running, execute:
   ```bash
   ps aux | grep python
   ```

2. **Monitor Logs:**
   View the logs in real time using:
   ```bash
   tail -f output_training.log
   tail -f output_classifier.log
   tail -f output_testing.log
   ```

3. **Stop Processes:**
   To stop any running process, locate its process ID (PID) with `ps aux` and then terminate it:
   ```bash
   kill <PID>
   ```

## Project Directory Structure

```plaintext
AutoMLClustering/
├── config.sh                     # Configuration script to set up the environment
├── datasets/                     # Contains datasets for training and testing
│   ├── train/                    # Training datasets
│   │   ├── beers                 # Beers dataset
│   │   ├── flights               # Flights dataset
│   │   ├── hospital              # Hospital dataset
│   │   ├── rayyan                # Rayyan dataset
│   │   └── ...                   # Other datasets (if applicable)
│   └── test/                     # Placeholder for testing datasets
├── LICENSE                       # Project license (e.g., MIT License)
├── README.md                     # Project documentation
├── results/                      # Directory to store results (e.g., logs, outputs)
├── src/                          # Source code for the project
│   ├── cleaning/                 # Data cleaning modules
│   │   ├── baran                 # Baran cleaning algorithm
│   │   ├── mode                  # Mode cleaning algorithm
│   │   └── ...                   # Other cleaning algorithms (if applicable)
│   ├── clustering/               # Clustering methods
│   │   ├── DBSCAN                # DBSCAN clustering
│   │   ├── GMM                   # Gaussian Mixture Model clustering
│   │   ├── HC                    # Hierarchical clustering
│   │   ├── KMEANS                # K-Means clustering
│   │   └── ...                   # Other clustering algorithms (if applicable)
│   ├── pipeline/                 # Pipeline implementation
│   │   ├── train/                # Training pipeline
│   │   │   ├── pre-processing.py         # Preprocessing routines for training data
│   │   │   ├── train_pipeline.py         # Main script orchestrating the training pipeline
│   │   │   ├── classifier.py             # Classification logic and training code
│   │   │   ├── classifier_preparation.py # Prepares data for classifier training
│   │   │   ├── cluster_methods.py        # Clustering utility functions and methods
│   │   │   ├── clustered_analysis.py     # Analyzes clustering results and metrics
│   │   │   └── error_correction.py       # Error correction module for training phase
│   │   └── test/                 # Testing pipeline
```

## Logs and Outputs

- Logs are saved in `output_training.log`, `output_classifier.log` and `output_testing.log`
- Intermediate results and analysis outputs are stored in the `results` directory.

## Contributing

1. Fork the repository.
2. Create your feature branch: `git checkout -b feature/YourFeature`.
3. Commit your changes: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature/YourFeature`.
5. Submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.