<p align="center">
  <img src="logo_main.png" alt="X-Sieve Logo" width="300"/>
</p>


🧪 X-Sieve: XIAP Screening and Inhibitor Evaluation Engine
X-Sieve is a Streamlit-based application designed for the virtual screening of natural product-based molecules targeting XIAP (X-linked inhibitor of apoptosis protein). It leverages six robust machine learning classifiers to predict potential bioactivity, alongside key ADME properties for drug-likeness evaluation.
🚀 Features
Single Molecule Prediction
Enter a SMILES string and get predictions on XIAP inhibitory activity using:

Random Forest (RF)
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
XGBoost
LightGBM
Extra Trees
Batch Mode Analysis

Upload a CSV file containing SMILES (single column, heading SMILES) and receive:
Bioactivity predictions from all six classifiers
Molecular property calculations (MW, LogP, HBA, HBD, Rotatable Bonds, TPSA)
Lipinski rule evaluation (number of violations)

Interactive Interface
Built with Streamlit, the app provides an intuitive UI suitable for both beginners and experienced users in cheminformatics and drug discovery.

📂 Input Format
🔹 Single Molecule Mode:
Input: SMILES string (e.g., CN1C=NC2=C1C(=O)N(C(=O)N2C)C)

🔹 Batch Mode:
Input file: CSV
Requirements:
One column only
Header name: SMILES

Example:
mathematica
Copy
Edit
SMILES
CC(=O)OC1=CC=CC=C1C(=O)O
COC1=CC=CC=C1OC
📊 Output
Bioactivity: "Active" or "Inactive" prediction per model, with probabilities
Drug-likeness: Lipinski violation count
Molecular properties: MW, LogP, HBA, HBD, TPSA, Rotatable Bonds

--Developers
Sheikh Sunzid Ahmed
M. Oliur Rahman
Plant Taxonomy and Ethnobotany Laboratory
Department of Botany, University of Dhaka
