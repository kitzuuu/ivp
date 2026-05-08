# IVP Hindi Digit Classification

The dataset contains 32x32 grayscale images of Hindi digits. A compact CNN is a good fit here because the task is image-based, the labels are balanced and clean, and the model can learn local stroke patterns directly from pixels.

## Train and Create Submission

Use the project virtual environment, then run:

```powershell
& ".\.venv\Scripts\python.exe" train_cnn.py --epochs 25 --batch-size 512 --device cuda --submission-path submission.csv --model-path best_cnn.pt
```

For a stronger submission, train several seeds and average them. This usually improves leaderboard accuracy because the remaining mistakes are mostly model-variance errors:

```powershell
New-Item -ItemType Directory -Force -Path models
& ".\.venv\Scripts\python.exe" train_cnn.py --epochs 30 --batch-size 512 --device cuda --seed 42  --model-path models\cnn_seed42.pt  --submission-path submission_seed42.csv
& ".\.venv\Scripts\python.exe" train_cnn.py --epochs 30 --batch-size 512 --device cuda --seed 123 --model-path models\cnn_seed123.pt --submission-path submission_seed123.csv
& ".\.venv\Scripts\python.exe" train_cnn.py --epochs 30 --batch-size 512 --device cuda --seed 777 --model-path models\cnn_seed777.pt --submission-path submission_seed777.csv
& ".\.venv\Scripts\python.exe" train_cnn.py --skip-training --batch-size 512 --device cuda --model-paths models\cnn_seed42.pt models\cnn_seed123.pt models\cnn_seed777.pt --submission-path submission.csv
```

To regenerate `submission.csv` from the saved model without retraining:

```powershell
& ".\.venv\Scripts\python.exe" train_cnn.py --skip-training --batch-size 512 --device cuda --submission-path submission.csv --model-path best_cnn.pt
```

The submission file format is:

```csv
Id,Category
56604,6
29396,3
```
