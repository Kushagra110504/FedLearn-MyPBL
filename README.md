🚀 Federated Learning IDS with Chimp Optimization

A distributed Intrusion Detection System (IDS) leveraging Federated Learning (FL), Chimp Optimization Algorithm (ChOA), and Deep Learning (1D-CNN) on the CICIDS2017 dataset.

📌 Overview

This project implements a privacy-preserving IDS using a federated learning architecture combined with meta-heuristic optimization for intelligent feature selection.

🔹 Key Highlights

🌐 Federated Learning (FedAvg) for distributed model training

🐒 Chimp Optimization Algorithm (ChOA) for automated feature selection

🧠 1D Convolutional Neural Network (1D-CNN) for attack classification

🔒 Privacy-preserving decentralized training

📊 Streamlit dashboard for visualization & model evaluation

📈 Comprehensive evaluation (Accuracy, Precision, Recall, F1-Score, Confusion Matrix)

📂 Dataset

CICIDS2017 – Network Traffic Classification Dataset

You can download it from:

🔗 UNB Website:
https://www.unb.ca/cic/datasets/ids-2017.html

🔗 Kaggle:
https://www.kaggle.com/datasets

📌 Required File

After downloading:

Extract the CSV files

Rename the file to:

CICIDS2017.csv


Place it inside:

data/

🏗️ Project Structure
FL_IDS_CICIDS/
│
├── main_training_pipeline.py
├── run_frontend.py
├── requirements.txt
│
├── data/
│   └── CICIDS2017.csv
│
├── models/
│   └── best_global_model.pth
│
├── checkpoints/
│   └── global_round_*.pth
│
├── evaluation/
│   ├── training_history.csv
│   └── evaluation_results.csv
│
├── federated/
│   ├── server.py
│   └── model.py
│
├── chimp_optimization/
│   ├── choa.py
│   └── choa_convergence.py
│
├── cnn_model/
│   └── model.py
│
├── preprocessing/
│   ├── data_loader.py
│   ├── feature_selection.py
│   └── data_splitter.py
│
├── utils/
│   └── metrics.py
│
└── app.py


⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/yourusername/FL_IDS_CICIDS.git
cd FL_IDS_CICIDS

2️⃣ Install Dependencies
pip install -r requirements.txt

🧠 How It Works
Step 1 – Preprocessing

Data cleaning

Label encoding

Feature scaling

Train-test split

Step 2 – Feature Selection

Chimp Optimization Algorithm selects the most relevant features to:

Reduce dimensionality

Improve training efficiency

Avoid overfitting

Step 3 – Federated Training

Multiple clients train locally

FedAvg aggregates weights

Global model updated iteratively

Step 4 – Evaluation

Metrics used:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

Loss convergence

All metrics saved to:

evaluation/training_history.csv

▶️ Running the Project
🔹 Train the Model
python main_training_pipeline.py


Outputs:

Selected feature indices

Best global model checkpoint

Training history

🔹 Launch Dashboard (Streamlit)
streamlit run app.py


OR

python run_frontend.py


The dashboard allows you to:

Visualize training progress

View evaluation metrics

Test trained model

🧪 Configuration

You can modify:

ChOA parameters → chimp_optimization/choa.py

CNN architecture → cnn_model/model.py

Federated client settings → federated/server.py

Number of rounds & clients → main_training_pipeline.py

📊 Performance Metrics

The system evaluates using standard classification metrics:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Loss convergence across federated rounds

Results are visualized in the Streamlit dashboard.

🔐 Why Federated Learning?

✔ No raw data leaves client
✔ Enhanced privacy
✔ Scalable distributed training
✔ Real-world deployable IDS framework

🏁 Future Improvements

Differential Privacy integration

Secure aggregation

Adaptive client weighting

Real-time traffic streaming support

👨‍💻 Author

Mentor- Dr. Neha Janu
Kushagra Singh-2427030078
B.Tech – CSE
Federated Learning Research Project


