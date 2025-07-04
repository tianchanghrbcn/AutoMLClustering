# AutoMLClustering

This project implements an automated machine learning pipeline for clustering tasks, supporting various data preprocessing, error correction, clustering methods, and result analysis.

## Prerequisites

- Python 3.9 or above
- Linux-based system (tested on Ubuntu)
- Required packages (installed via `config.sh`)

## Installation

1. **Clone the repository:**

   ```bash
   git clone git@github.com:tianchanghrbcn/AutoMLClustering.git
   cd /root/AutoMLClustering
   ```

2. **Run the configuration script** to set up the virtual environment and install dependencies:

   ```bash
   bash config.sh
   ```

3. **Activate the virtual environment:**

   ```bash
   conda activate torch110
   ```

4. **Set up the `PYTHONPATH` to avoid `ModuleNotFoundError` (Optional):**  
   If you encounter `ModuleNotFoundError` when running the training pipeline, you may need to permanently add the project directory to your `PYTHONPATH`:

   ```bash
   nano ~/.bashrc
   ```
   Add the following line at the end of the file:
   ```bash
   export PYTHONPATH=/root/AutoMLClustering
   ```
   Save the file (`Ctrl + X`, then `Y`, then `Enter`), and apply the changes by running:
   ```bash
   source ~/.bashrc
   ```
   If the issue persists, you may need to restart your terminal or system.
   
## Running the Project

### Step 1: Data Preprocessing
Navigate to the `src/pipeline/train` directory and run the preprocessing script:

```bash
cd src/pipeline/train
python pre-processing.py
```

### Step 2: Start the Training Pipeline

Now, start the training pipeline in the background using `nohup` to ensure it continues running even if the session is disconnected:

```bash
nohup python train_pipeline.py > output_training.log 2>&1 &
```

- **Note:** The `PYTHONPATH` variable is now set permanently and will persist across sessions.
- The output and logs will be saved to the `output_training.log` file.

### Step 3: Start the Classifier
After completing the training pipeline, run the classifier in the background using the `nohup` mechanism:

```bash
nohup python classifier.py > output_classifier.log 2>&1 &
```

- **Note:** Ensure that the classifier script is in the correct directory and has all the required configurations.
- Logs and outputs for the classifier will be stored in the `output_classifier.log` file.

### Step 4: Run the Test Pipeline
Once the training pipeline has been executed, navigate to the `src/pipeline/test` directory and run the test pipeline:

```bash
cd src/pipeline/test
nohup python test_pipeline.py > output_testing.log 2>&1 &
```

- This step executes the testing pipeline to validate the training results.

### Step 5: Compute Loss and Accuracy
After both the training and testing pipelines have completed, run the following script in the `src/pipeline/test` directory to compute the loss and accuracy metrics:

```bash
python compute_loss_and_acc.py
```

- This script analyzes the clustering results and computes the loss and accuracy.

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
├── references/                   # References and supporting documentation
├── requirements.txt              # Python dependencies
├── results/                      # Directory to store results (e.g., logs, outputs)
├── src/                          # Source code for the project
│   ├── cleaning/                 # Data cleaning modules
│   │   ├── baran                 # Baran cleaning algorithm
│   │   ├── mode                  # Mode cleaning algorithm
│   │   └── ...                   # Other cleaning algorithms (if applicable)
│   ├── clustering/               # Clustering methods
│   │   ├── AP                    # Affinity Propagation clustering
│   │   ├── DBSCAN                # DBSCAN clustering
│   │   ├── GMM                   # Gaussian Mixture Model clustering
│   │   ├── HC                    # Hierarchical clustering
│   │   ├── KMEANS                # K-Means clustering
│   │   ├── OPTICS                # OPTICS clustering
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
│   │       ├── function_back.py          # Maps predictions back to specific strategies
│   │       ├── test_classify.py          # Testing classification: loads model and evaluates test data
│   │       ├── test_error_correction.py  # Error correction module for testing phase
│   │       ├── compute_loss_and_acc.py   # Computes loss/accuracy for testing tasks
│   │       ├── pre_processing_test.py    # Preprocessing routines specific to test data
│   │       ├── test_analysis.py          # Analyzes test clustering results
│   │       └── test_pipeline.py          # Main script orchestrating the test pipeline
│   └── --legacy/                 # Legacy or deprecated code
├── venv/                         # Virtual environment directory (created by config.sh)
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