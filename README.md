# Solar Panel Defect Detection Using Deep Learning

## Project Overview
This project detects and classifies solar panel defects using deep learning models trained on the ELPV electroluminescence (EL) dataset.

## Dataset
- Total images: 2,624
- Image dimensions: 300 x 300
- Image type: Grayscale
- Classes: 0, 0.33, 0.66, 1.0
- Training images: 1,836
- Validation images: 394
- Testing images: 394

## Models Implemented
1. Baseline CNN
2. Balanced CNN
3. ResNet18
4. ResNet18 V2

## Test Results

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Macro ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline CNN | 68.27% | 0.3327 | 0.3778 | 0.3524 | 0.6997 |
| Balanced CNN | 64.97% | 0.3611 | 0.4012 | 0.3768 | Not available |
| ResNet18 | 48.22% | 0.4530 | 0.4648 | 0.4101 | 0.7351 |
| ResNet18 V2 | 35.53% | 0.4210 | 0.4541 | 0.3361 | 0.7536 |

## Project Structure

- src/: Dataset preparation, model training and evaluation scripts
- results/: Evaluation metrics, classification reports and visualizations
- models/: Trained model weights (not included in this repository)
- data/: Original dataset (not included in this repository)

## Technologies Used
- Python
- PyTorch
- CNN
- ResNet18
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Limitations
- The dataset is relatively small.
- Performance varies across defect classes.
- The trained model weights and original dataset are not included in this repository.
- Further improvements are needed before real-world deployment.

## Future Improvements
- Improve minority-class classification.
- Experiment with data augmentation and transfer learning.
- Evaluate models on additional solar panel datasets.
- Explore defect localization.

## Author
Saritha
