import sys
import subprocess
# Install system dependencies for RDKit
subprocess.run(["apt-get", "update"], stdout=sys.stdout, stderr=sys.stderr)
subprocess.run(["apt-get", "install", "-y", "libxrender1", "libxext6"], stdout=sys.stdout, stderr=sys.stderr)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, Lipinski
from PIL import Image
import os
from sklearn.preprocessing import StandardScaler

# Configure page
st.set_page_config(page_title="X-Sieve", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        .header {
            background-color: #2e7d32;
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
        }
        .model-card {
            background: white;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .property-card {
            background: #e8f5e9;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .error-card {
            background: #ffebee;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header">
        <h1 style="margin:0; text-align:center">X-Sieve</h1>
        <p style="margin:0; text-align:center">XIAP Screening and Inhibitor Evaluation Engine</p>
    </div>
""", unsafe_allow_html=True)

# Load models and feature selector
@st.cache_resource
def load_resources():
    resources = {
        "models": {},
        "scaler": None,
        "feature_indices": None
    }
    
    # Load feature indices with validation
    try:
        if os.path.exists("selected_feature_indices.npy"):
            feature_indices = np.load("selected_feature_indices.npy")
            if len(feature_indices) == 100:
                resources["feature_indices"] = feature_indices
                st.sidebar.success("✓ Feature indices loaded (100 features)")
            else:
                st.sidebar.error(f"✗ Feature indices: Expected 100, got {len(feature_indices)}")
        else:
            st.sidebar.error("✗ Feature indices file not found")
    except Exception as e:
        st.sidebar.error(f"Error loading feature indices: {str(e)}")
    
    # Load models (removed Logistic Regression, Naive Bayes, and MLP)
    model_files = {
        "Random Forest": "RF.pkl",
        "SVM": "SVM.pkl", 
        "KNN": "KNN.pkl",
        "XGBoost": "XGBoost.pkl",
        "LightGBM": "LightGBM.pkl",
        "Extra Trees": "ET.pkl"
    }
    
    for name, file in model_files.items():
        try:
            if os.path.exists(file):
                resources["models"][name] = joblib.load(file)
                st.sidebar.success(f"✓ {name} loaded")
            else:
                st.sidebar.error(f"✗ {name}: File not found")
        except Exception as e:
            st.sidebar.error(f"✗ {name} failed to load: {str(e)}")
    
    # Load scaler if exists (kept in case other models need it)
    try:
        if os.path.exists("scaler.pkl"):
            resources["scaler"] = joblib.load("scaler.pkl")
            st.sidebar.success("✓ Scaler loaded")
    except Exception as e:
        st.sidebar.error(f"Error loading scaler: {str(e)}")
    
    return resources

# Sidebar
with st.sidebar:
    st.header("System Status")
    resources = load_resources()
    
    st.markdown("---")
    st.header("Developers")
    st.markdown("""
    Sheikh Sunzid Ahmed  
    M. Oliur Rahman
    """)
    
    st.markdown("---")
    st.header("Instructions")
    st.markdown("""
    1. Enter SMILES or upload CSV
    2. View predictions and properties
    3. Download results
    """)

# Helper functions
def smiles_to_features(smiles, feature_indices=None):
    """Convert SMILES to features using Morgan fingerprints and apply feature selection"""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    
    # Generate 2048-bit fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    features = np.array(fp)
    
    # Apply feature selection if indices are provided
    if feature_indices is not None:
        try:
            features = features[feature_indices]
            if len(features) != 100:
                st.error(f"Feature selection error: Expected 100 features, got {len(features)}")
                return None
        except IndexError as e:
            st.error(f"Feature selection error: {str(e)}")
            return None
    
    return features

def calculate_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    
    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "RotBonds": Lipinski.NumRotatableBonds(mol),
        "TPSA": Descriptors.TPSA(mol),
        "Lipinski_Violations": sum([
            Descriptors.MolWt(mol) > 500,
            Descriptors.MolLogP(mol) > 5,
            Lipinski.NumHAcceptors(mol) > 10,
            Lipinski.NumHDonors(mol) > 5
        ])
    }

# Main app
tab1, tab2 = st.tabs(["Single Molecule", "Batch Processing"])

with tab1:
    st.header("Single Molecule Analysis")
    smiles = st.text_input("Enter SMILES string:", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    
    if st.button("Analyze"):
        if not smiles:
            st.warning("Please enter a SMILES string")
            st.stop()
            
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            st.error("Invalid SMILES string")
            st.stop()
            
        # Display molecule
        img = Draw.MolToImage(mol, size=(400, 300))
        st.image(img, caption="Molecular Structure")
        
        # Calculate properties
        props = calculate_properties(smiles)
        if props:
            st.subheader("Molecular Properties")
            st.markdown(f"""
            <div class="property-card">
                <b>Molecular Weight:</b> {props['MW']:.2f} {"(≤500)" if props['MW'] <= 500 else "(>500)"}<br>
                <b>LogP:</b> {props['LogP']:.2f} {"(≤5)" if props['LogP'] <= 5 else "(>5)"}<br>
                <b>H-bond Acceptors:</b> {props['HBA']} {"(≤10)" if props['HBA'] <= 10 else "(>10)"}<br>
                <b>H-bond Donors:</b> {props['HBD']} {"(≤5)" if props['HBD'] <= 5 else "(>5)"}<br>
                <b>Rotatable Bonds:</b> {props['RotBonds']}<br>
                <b>TPSA:</b> {props['TPSA']:.2f}
            </div>
            """, unsafe_allow_html=True)
            
            # Lipinski evaluation
            if props["Lipinski_Violations"] == 0:
                st.success("✅ Zero Lipinski violations (good drug-like properties)")
            else:
                st.warning(f"⚠️ {props['Lipinski_Violations']} Lipinski violation(s)")
        
        # Make predictions
        features = smiles_to_features(smiles, resources["feature_indices"])
        if features is None:
            st.error("Could not generate valid features for this molecule")
            st.stop()
            
        st.subheader("Model Predictions")
        
        if resources["feature_indices"] is None:
            st.error("Feature selection not available - cannot make predictions")
            st.stop()
            
        for name, model in resources["models"].items():
            try:
                # Prepare features
                X = features.reshape(1, -1)
                
                # Make prediction
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)[0][1]
                else:
                    proba = float(model.predict(X)[0])
                
                pred = "Active" if proba >= 0.5 else "Inactive"
                
                st.markdown(f"""
                <div class="model-card">
                    <b>{name}:</b> <span style='color:{"green" if pred == "Active" else "red"}'>{pred}</span>
                    <br>Probability: {proba:.4f}
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f"""
                <div class="error-card">
                    <b>{name}:</b> Error in prediction<br>
                    <small>{str(e)}</small>
                </div>
                """, unsafe_allow_html=True)

with tab2:
    st.header("Batch Processing")
    uploaded_file = st.file_uploader("Upload CSV file with SMILES column", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "SMILES" not in df.columns:
            st.error("CSV must contain a 'SMILES' column")
            st.stop()
            
        if resources["feature_indices"] is None:
            st.error("Cannot process batch - feature selection not available")
            st.stop()
            
        results = []
        invalid_smiles = []
        error_molecules = []
        
        with st.spinner("Processing molecules..."):
            progress_bar = st.progress(0)
            
            for i, smi in enumerate(df["SMILES"]):
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if not mol:
                        invalid_smiles.append(i+2)  # +2 for header and 1-based index
                        continue
                        
                    # Calculate properties
                    props = calculate_properties(smi)
                    if not props:
                        error_molecules.append((i+2, "Property calculation failed"))
                        continue
                        
                    # Generate features
                    features = smiles_to_features(smi, resources["feature_indices"])
                    if features is None:
                        error_molecules.append((i+2, "Feature generation failed"))
                        continue
                        
                    # Make predictions
                    preds = {"SMILES": smi}
                    
                    for name, model in resources["models"].items():
                        try:
                            X = features.reshape(1, -1)
                            
                            if hasattr(model, "predict_proba"):
                                proba = model.predict_proba(X)[0][1]
                            else:
                                proba = float(model.predict(X)[0])
                                
                            preds[f"{name}_Pred"] = "Active" if proba >= 0.5 else "Inactive"
                            preds[f"{name}_Prob"] = proba
                        except Exception as e:
                            preds[f"{name}_Pred"] = "Error"
                            preds[f"{name}_Prob"] = np.nan
                            error_molecules.append((i+2, f"{name} prediction error: {str(e)}"))
                    
                    # Add properties
                    preds.update({
                        "MW": props["MW"],
                        "LogP": props["LogP"],
                        "HBA": props["HBA"],
                        "HBD": props["HBD"],
                        "RotBonds": props["RotBonds"],
                        "TPSA": props["TPSA"],
                        "Lipinski_Violations": props["Lipinski_Violations"]
                    })
                    
                    results.append(preds)
                except Exception as e:
                    error_molecules.append((i+2, f"General processing error: {str(e)}"))
                
                progress_bar.progress((i+1)/len(df))
        
        # Show results
        if results:
            results_df = pd.DataFrame(results)
            st.success(f"Processed {len(results)} valid molecules")
            st.dataframe(results_df)
            
            # Download button
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Results",
                csv,
                "x-sieve_results.csv",
                "text/csv"
            )
        
        if invalid_smiles:
            st.warning(f"Skipped {len(invalid_smiles)} invalid SMILES (rows: {', '.join(map(str, invalid_smiles))})")
        
        if error_molecules:
            st.warning(f"Encountered {len(error_molecules)} processing errors")
            if st.checkbox("Show error details"):
                for row, error in error_molecules:
                    st.write(f"Row {row}: {error}")

# Footer with institution name
st.markdown("""
    <div style="text-align: center; color: #2e7d32; margin-top: 2rem;">
        Department of Botany, University of Dhaka
    </div>
""", unsafe_allow_html=True)
