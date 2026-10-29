# Dashcam-Based Vehicle Speed Estimation
Project 1 for AIT 790

Project overview
This project investigates estimating a vehicle's speed in miles per hour (mph) from dashcam video. The intended system processes video frames or short frame sequences and produces a speed estimate suitable for real-time or near-real-time use.

The current candidate architecture is a two-stage pipeline:
1. Movement classifier — determines whether the vehicle is stationary or moving.
2. Speed regressor — estimates speed in mph when movement is detected.
The project uses a stationary threshold of 2 mph or less. The architecture is a candidate design, not a claim that a trained model or validated performance result already exists.
Goals
- Estimate vehicle speed from dashcam video.
- Identify stationary periods using the project's threshold (speed <= 2 mph).
- Evaluate speed-estimation error against reference/ground-truth speeds.
- Evaluate stationary-versus-moving classification.
- Track inference latency so accuracy can be considered alongside real-time feasibility.
Suggested project structure
dashcam-speed-estimation/
├── dashcam_speed_architecture_overleaf.tex  # Research/architecture paper
├── README.md                                # This file
├── evaluate_accuracy.py                     # Independent evaluation script
├── data/
│   └── predictions.csv                      # Ground truth and model outputs
└── results/
    └── evaluation_report.json               # Generated metrics (optional)
Evaluation data format
Prepare a CSV with one row per evaluated frame or time sample. The evaluation script expects:
Column	Meaning
ground_truth_mph	Reference speed in mph
predicted_mph	Model-estimated speed in mph
latency_ms (optional)	End-to-end processing latency for that sample, in milliseconds

Metrics
The independent evaluation script reports:
Speed regression
- MAE — mean absolute error in mph.
- RMSE — root mean squared error in mph, which penalizes larger errors more strongly.
- Bias — mean(predicted speed - ground-truth speed); positive values indicate overestimation on average.
- R² — coefficient of determination when the ground-truth values permit a valid calculation.
Stationary/moving classification
Ground-truth and predicted labels are derived using the same threshold: speed <= 2 mph is stationary; speed > 2 mph is moving. The report includes:
- Accuracy
- Precision, recall, and F1 for the moving class
- Confusion-matrix counts (true stationary, false moving, missed moving, correctly moving)
