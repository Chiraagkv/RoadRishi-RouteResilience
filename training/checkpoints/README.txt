Place the trained model checkpoint here.

Expected filename: roadrishi_finetuned.pth   (the best checkpoint written by the training script)

Steps:
1. Train with training/kaggle_segformer_train.py
   - Kaggle: GPU T4 x2, attach DeepGlobe, use "Save Version -> Save & Run All"
   - or locally: see README.md ("Training")
2. Download from the output folder: roadrishi_finetuned.pth, metrics.json, history.json,
   curves.png, predictions_grid.png (metrics.json is the source of truth for reported numbers)
3. Put roadrishi_finetuned.pth in this folder.

The backend then switches /api/checkpoint-status from "simulation" to "live_model" automatically.
(*.pth files are gitignored — the checkpoint is not committed.)
