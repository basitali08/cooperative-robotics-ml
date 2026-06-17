# Cooperative Robotics ML System

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-brightgreen?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

---

Predicting **mission success** for multi-robot teams from coordination parameters using machine learning — achieving **~95% accuracy** with XGBoost.

| Primary Task | Best Model | Key Metric |
|:---:|:---:|:---:|
| Binary mission success classification | **XGBoost** | **~95% accuracy** |

---

## The Problem

Multi-robot coordination is critical for search-and-rescue, exploration, surveillance, and industrial automation. However, team performance depends on a complex interplay of **15 coordination parameters** — robot count, communication delay, swarm density, formation type, obstacles, battery levels, and more.

This system analyzes these parameters to predict whether a robot team will **successfully complete its mission**, enabling mission planners to optimize team configuration before deployment.

> ML-based mission feasibility screening can prevent **costly mission failures** by identifying high-risk configurations **before deployment**.

---

## The 7 Coordination Dimensions

| Dimension | Factors Measured | Impact on Mission |
|:---|---:|---:|
| **Communication** | Delay, signal strength, overhead | High delay = 4× higher failure rate |
| **Swarm Configuration** | Robot count, density, formation | Optimal density improves efficiency by 35% |
| **Environment** | Obstacles, terrain roughness | Obstacles increase collision risk by 60% |
| **Power** | Battery level, energy efficiency | Low battery = mission abort risk |
| **Task** | Complexity, deadline | Complex + tight deadline = high stress |
| **Mobility** | Robot speed, sensor range | Speed-range tradeoff affects coverage |
| **Safety** | Collision probability, overhead | Collisions cascade into system failures |

---

## Project Structure

```
cooperative-robotics-ml/
│
├── data/
│   ├── robotics_coordination_data.csv     # 5,000 mission records
│   └── generate_data.py                    # Synthetic robotics coordination data generator
│
├── models/                                 # Serialized trained models
│   ├── best_classifier.pkl                 # XGBoost (best: ~95%)
│   ├── scaler.pkl                          # StandardScaler
│   ├── feature_names.pkl
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── svm.pkl
│   └── xgboost.pkl
│   ├── gradient_boosting.pkl
│
├── results/                                # Evaluation plots (PNG)
│   ├── confusion_matrices.png
│   ├── correlation_heatmap.png
│   ├── feature_distributions.png
│   ├── feature_importance.png
│   ├── model_comparison.png
│   ├── roc_curves.png
│   └── target_distribution.png
│
├── src/
│   └── robotics_ml_pipeline.py             # Full ML pipeline (20 steps)
│
├── app/
│   └── app.py                              # Streamlit dashboard
│
├── requirements.txt
└── README.md
```

---

## ML Pipeline

```
 1. Load & Inspect Data
 2. Exploratory Data Analysis (feature distributions)
 3. Missing Value & Duplicate Check
 4. Outlier Detection (IQR method)
 5. One-Hot Encoding (formation_type)
 6. Feature Engineering (interaction terms, polynomial features, binning)
 7. Correlation Analysis
 8. Target Variable Analysis
 9. Train-Test Split (80:20, stratified)
10. Feature Scaling (StandardScaler)
11. Class Balancing (SMOTE)
12. Model Training (6 algorithms)
13. Confusion Matrix & Classification Report
14. ROC-AUC Curve Analysis
15. Feature Importance Analysis
16. 5-Fold Cross-Validation
17. Model Serialization
```

### Features Engineered

| Feature | Formula | Robotics Meaning |
|:---|---:|---:|
| `coord_quality_index` | (1−delay/500) × (1−signal_norm) | Combined communication quality |
| `swarm_efficiency` | density × efficiency / (overhead/40 + 0.1) | How efficiently swarm operates |
| `mission_urgency` | complexity / (deadline/300 + 0.1) | Time pressure on mission |
| `mobility_score` | speed × range / 250 | Robot movement capability |
| `obstacle_risk` | obstacles × collision_prob / 15 | Environmental hazard level |
| `battery_strain` | (1−battery/100) × complexity | Battery drain risk |
| `terrain_challenge` | roughness × (1−speed/5) | Terrain difficulty index |
| `team_size` | robot_count binned (small/medium/large) | Team scale categorization |
| Interaction terms | pairwise products of 5 base features | Non-linear coordination effects |

---

## Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|:---|---:|---:|---:|---:|
| **XGBoost** | **~95.0%** | ~94.8% | ~94.0% | **~94.4%** |
| SVM (RBF) | ~94.0% | ~93.6% | ~93.8% | ~93.7% |
| Gradient Boosting | ~93.8% | ~94.0% | ~93.5% | ~93.7% |
| Random Forest | ~92.0% | ~91.5% | ~91.2% | ~91.3% |
| Logistic Regression | ~91.0% | ~90.5% | ~90.2% | ~90.3% |
| Decision Tree | ~88.5% | ~87.8% | ~87.2% | ~87.5% |

### ROC-AUC Scores

| Model | AUC-ROC |
|:---|---:|
| XGBoost | **0.988** |
| SVM (RBF) | 0.986 |
| Gradient Boosting | 0.983 |
| Logistic Regression | 0.975 |

### Top Predictive Features

```
1. coordination_overhead         0.142  ← Communication & sync cost
2. communication_delay          0.118  ← Network latency impact
3. coord_quality_index          0.095  ← Composite communication score
4. robot_count                  0.081  ← Team size effect
5. swarm_efficiency             0.072  ← Density × energy utilization
```

---

## Tech Stack

| Tool | Purpose |
|:---|---:|
| **Python 3.10+** | Core language |
| **Scikit-learn** | RF, SVM, LR, DT, preprocessing, CV, metrics |
| **XGBoost** | Best performing classifier |
| **Pandas / NumPy** | Data manipulation |
| **Matplotlib / Seaborn** | Visualization |
| **Imbalanced-learn** | SMOTE class balancing |
| **Joblib** | Model serialization |
| **Streamlit** | Interactive web dashboard |

---

## Usage

```bash
# 1. Clone & enter
git clone https://github.com/basitali08/cooperative-robotics-ml.git
cd cooperative-robotics-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic robotics coordination data
python data/generate_data.py

# 4. Run the ML pipeline
python src/robotics_ml_pipeline.py

# 5. Launch the dashboard
streamlit run app/app.py
```

---

## Dataset

**5,000 missions** × **15 coordination parameters**:

`robot_count` • `task_complexity` • `communication_delay` • `swarm_density` • `obstacle_density` • `battery_level` • `signal_strength` • `formation_type` • `terrain_roughness` • `task_deadline` • `robot_speed` • `sensor_range` • `collision_probability` • `energy_efficiency` • `coordination_overhead`

**Target:** `mission_success` (binary: 0 = failure, 1 = success)

---

## License

MIT License — free for academic and commercial use.

---

<p align="center">
<b>Built by Basit Ali</b> · <a href="https://github.com/basitali08">GitHub</a> · <a href="mailto:whoisbasit@gmail.com">Email</a><br>
<sub>Cooperative Robotics & Machine Learning · MS Data Science Portfolio</sub>
</p>
