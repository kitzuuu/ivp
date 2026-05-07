# IVP Hindi Digit Classification

The dataset contains 32x32 grayscale images of Hindi digits. A compact CNN is a good fit here because the task is image-based, the labels are balanced and clean, and the model can learn local stroke patterns directly from pixels.

## Train and Create Submission

Use the project virtual environment, then run:

```powershell
& ".\.venv\Scripts\python.exe" train_cnn.py --epochs 25 --batch-size 512 --device cuda --submission-path submission.csv --model-path best_cnn.pt
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
