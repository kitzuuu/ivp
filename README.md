# IVP Hindi digit classification project

Antonia Maria Constantin i6380645
Toma Cristian Nitu i6350367
Vlad Stoica i6406968

We used a compact CNN because the task is image-based, the labels are balanced and clean, and the model can learn local stroke patterns directly from pixels.

Make sure to have the virtual environment running, then run:

```powershell
& ".\.venv\Scripts\python.exe" train_cnn.py --epochs 25 --batch-size 512 --device cuda --submission-path submission.csv --model-path best_cnn.pt
```
