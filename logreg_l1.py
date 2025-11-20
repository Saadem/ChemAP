import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski, AllChem

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import joblib

# Création du sous-répertoire
SAVE_DIR = "./dataset/data_log"
os.makedirs(SAVE_DIR, exist_ok=True)

SEED = 42
# Chemins # jeu de référence
REF_CSV  = "dataset/DrugApp/All_training_feature_vectors.csv"   
# Chemins # approuvés FDA (SMILES)
FDA_CSV  = "dataset/FDA/FDA_2023_approved.csv"   
# Chemins # échecs ClinicalTrials (SMILES)
FAIL_CSV = "dataset/ClinicalTrials/clinical_fail_2024_05.csv"   
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Colonnes de descripteurs (continus) 
DESC_COLS = ["logP", "MW", "PSA", "HBA", "HBD", "Rings", "Refractivity", "Bond_#"]

def select_molecular_block(df):
    """Retourne la liste des colonnes moléculaires (descripteurs + ECFP4.*)."""
    ecfp_cols = sorted(
        [c for c in df.columns if c.startswith("ECFP4.")],
        key=lambda x: int(x.split(".")[1])
    )
    keep = [c for c in DESC_COLS if c in df.columns] + ecfp_cols
    return keep

def featurize_smiles_series(smiles_series, n_bits=128, radius=2):
    """SMILES -> DataFrame (descripteurs RDKit + bits ECFP4). Lignes invalides ignorées."""
    rows, idx_keep = [], []
    for idx, s in smiles_series.items():
        if pd.isna(s):
            continue
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            continue
        try:
            row = {
                "logP": Crippen.MolLogP(mol),
                "MW": Descriptors.MolWt(mol),
                "PSA": rdMolDescriptors.CalcTPSA(mol),
                "HBA": Lipinski.NumHAcceptors(mol),
                "HBD": Lipinski.NumHDonors(mol),
                "Rings": rdMolDescriptors.CalcNumRings(mol),
                "Refractivity": Crippen.MolMR(mol),
                "Bond_#": mol.GetNumBonds(),
            }
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
            bits = [int(b) for b in fp.ToBitString()]
            for i in range(1, n_bits + 1):
                row[f"ECFP4.{i}"] = bits[i - 1]
            rows.append(row)
            idx_keep.append(idx)
        except Exception:
            continue
    return pd.DataFrame(rows, index=idx_keep)

def enforce_dtypes(df, desc_cols, ecfp_prefix="ECFP4."):
    """Force les descripteurs en float64 et (optionnel) les bits ECFP en int8."""
    for c in desc_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    ecfp_cols = [c for c in df.columns if c.startswith(ecfp_prefix)]
    for c in ecfp_cols:
        # int8 = gain mémoire ; garde 0/1 nos bits
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int8")
    return df

def plot_and_save_roc_pr(y_true, y_proba, prefix):
    """Trace + enregistre ROC & PR. Retourne (AUROC, AUPRC, path_roc, path_pr)."""
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    # ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    #plt.title(f"ROC – {prefix}")
    plt.legend(loc="lower right")
    roc_path = os.path.join(RESULTS_DIR, f"roc_{prefix}.png")
    plt.savefig(roc_path, dpi=220)
    plt.show()

    # PR
    plt.figure()
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    # baseline horizontale = proportion de positifs
    baseline = y_true.mean()
    plt.hlines(baseline, 0, 1, colors="gray", linestyles="--",
               label=f"Baseline (pos={baseline:.2f})")
    plt.xlabel("Rappel")
    plt.ylabel("Précision")
    #plt.title(f"PR – {prefix}")
    plt.legend(loc="lower right")
    pr_path = os.path.join(RESULTS_DIR, f"pr_{prefix}.png")
    plt.savefig(pr_path, dpi=220)
    plt.show()

    return roc_auc, ap, roc_path, pr_path

def main():
    # RÉFÉRENCE 
    df_ref = pd.read_csv(REF_CSV)
    assert "Label" in df_ref.columns, "La colonne 'Label' est requise dans le jeu de référence."
    ref_cols = select_molecular_block(df_ref)
    df_ref = df_ref[["Label"] + ref_cols].dropna(axis=0, how="any")
    y_ref = df_ref["Label"].astype(int)
    X_ref = df_ref.drop(columns=["Label"])
    # float pour descripteurs, int8 pour ECFP
    X_ref = enforce_dtypes(X_ref, [c for c in DESC_COLS if c in X_ref.columns])

    # Pour df_ref (jeu de référence prétraité) 
    df_ref_pretraite = pd.concat([y_ref, X_ref], axis=1)
    df_ref_pretraite.to_csv(os.path.join(SAVE_DIR, "df_ref_pretraite.csv"), index=False)

    # Standardiser uniquement les descripteurs continus
    desc_to_scale = [c for c in DESC_COLS if c in X_ref.columns]
    scaler = StandardScaler()
    if desc_to_scale:
        # cast explicite en float avant transform (évite FutureWarning)
        X_ref.loc[:, desc_to_scale] = X_ref[desc_to_scale].astype("float64")
        X_ref.loc[:, desc_to_scale] = scaler.fit_transform(X_ref[desc_to_scale])

    # Sauvegarde schéma/scaler
    schema_path = os.path.join(RESULTS_DIR, "scaler_and_schema.joblib")
    joblib.dump(
        {"scaler": scaler, "desc_to_scale": desc_to_scale, "feature_order": X_ref.columns.tolist()},
        schema_path
    )

    # Split train/test
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_ref, y_ref, test_size=0.2, stratify=y_ref, random_state=SEED
    )

    # SÉLECTION DE VARIABLES (L1 avec CV)
    # Grille "plus pénalisante" pour favoriser une sélection plus parcimonieuse
    Cs_grid = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    l1_cv = LogisticRegressionCV(
        Cs=Cs_grid,
        cv=5,
        penalty="l1",
        solver="saga",
        scoring="roc_auc",
        class_weight="balanced",
        max_iter=5000,
        n_jobs=-1,
        refit=True,
        random_state=SEED
    )
    l1_cv.fit(X_tr, y_tr)

    # parametre optimal
    best_C = float(l1_cv.C_[0])  # C optimal (CV)
    print(f"[CV] C* (régularisation optimale) = {best_C:.6g}")

    # Sauvegarde du C* 
    pd.DataFrame({"metric": ["best_C"], "value": [best_C]}).to_csv(
    os.path.join(RESULTS_DIR, "model_selection_summary.csv"), index=False)
    
    coefs = pd.Series(l1_cv.coef_[0], index=X_tr.columns)
    selected_features = coefs[coefs != 0].index.tolist()
    if len(selected_features) == 0:
        selected_features = X_tr.columns.tolist()  # fallback très rare
    
    # AJOUTER juste après la construction de selected_features
    total_features = X_tr.shape[1]
    n_selected = len(selected_features)
    print(f"[Sélection] Variables retenues : {n_selected} / {total_features} "
      f"({n_selected/total_features:.1%})")

    # Append dans le même CSV de résumé (sans écraser la ligne best_C)
    summary_path = os.path.join(RESULTS_DIR, "model_selection_summary.csv")
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
    else:
        summary_df = pd.DataFrame(columns=["metric", "value"])

    summary_df = pd.concat([
        summary_df,
        pd.DataFrame({
        "metric": ["n_selected", "total_features", "selected_ratio"],
        "value": [n_selected, total_features, n_selected/total_features]
        })
    ], ignore_index=True)

    summary_df.to_csv(summary_path, index=False)



    pd.Series(selected_features, name="selected_features").to_csv(
        os.path.join(RESULTS_DIR, "selected_features_l1.csv"), index=False
    )

    # Ré-entraîner un modèle L1 final avec le C optimal sur les features retenues
    best_C = l1_cv.C_[0]
    model = LogisticRegression(
        penalty="l1", solver="saga", C=best_C,
        class_weight="balanced", max_iter=5000, random_state=42
    )
    model.fit(X_tr[selected_features], y_tr)

    joblib.dump(model, os.path.join(RESULTS_DIR, "logreg_l1_model.joblib"))
    joblib.dump(selected_features, os.path.join(RESULTS_DIR, "selected_features_list.joblib"))

    # ÉVAL – TEST RÉF
    proba_ref = model.predict_proba(X_te[selected_features])[:, 1]
    roc_ref, ap_ref, roc_ref_path, pr_ref_path = plot_and_save_roc_pr(
        y_te, proba_ref, prefix="ref_test_l1"
    )
    pd.DataFrame({"metric": ["AUROC", "AUPRC"], "value": [roc_ref, ap_ref]}).to_csv(
        os.path.join(RESULTS_DIR, "metrics_ref_test_l1.csv"), index=False
    )
    pd.DataFrame({"y_true": y_te.values, "y_proba": proba_ref}).to_csv(
        os.path.join(RESULTS_DIR, "predictions_ref_test_l1.csv"), index=False
    )

    # EXTERNE
    df_fda = pd.read_csv(FDA_CSV);  df_fda["Label"] = 1
    df_fail = pd.read_csv(FAIL_CSV); df_fail["Label"] = 0
    assert "SMILES" in df_fda.columns and "SMILES" in df_fail.columns, "SMILES manquants dans l'externe."

    feats_fda  = featurize_smiles_series(df_fda["SMILES"],  n_bits=128, radius=2)
    feats_fail = featurize_smiles_series(df_fail["SMILES"], n_bits=128, radius=2)

    fda_valid  = pd.concat([df_fda.loc[feats_fda.index,  ["Label"]], feats_fda],  axis=1)
    fail_valid = pd.concat([df_fail.loc[feats_fail.index, ["Label"]], feats_fail], axis=1)
    df_ext = pd.concat([fda_valid, fail_valid], ignore_index=True).dropna(axis=0, how="any")

    # Aligner avec le schéma du référentiel
    schema = joblib.load(schema_path)
    feature_order = schema["feature_order"]
    desc_to_scale = schema["desc_to_scale"]
    scaler = schema["scaler"]

    # Ajouter colonnes manquantes (bits ECFP absents -> 0 ; 
    # descripteurs absents -> 0.0)
    # (0 sur un bit = "motif absent" → pas une imputation de NA)
    for c in feature_order:
        if c not in df_ext.columns:
            if c in DESC_COLS:
                df_ext[c] = 0.0
            else:
                df_ext[c] = 0

    df_ext = df_ext[["Label"] + feature_order].dropna(axis=0, how="any")
    y_ext = df_ext["Label"].astype(int)
    X_ext = df_ext.drop(columns=["Label"])

    # Dtypes puis scaling des descripteurs
    X_ext = enforce_dtypes(X_ext, desc_to_scale)
    if desc_to_scale:
        X_ext.loc[:, desc_to_scale] = X_ext[desc_to_scale].astype("float64")
        X_ext.loc[:, desc_to_scale] = scaler.transform(X_ext[desc_to_scale])

    # Pour df_ext (jeu externe prétraité) 
    df_ext_pretraite = pd.concat([y_ext, X_ext], axis=1)
    df_ext_pretraite.to_csv(os.path.join(SAVE_DIR, "df_ext_pretraite.csv"), index=False)

    # Restreindre aux variables sélectionnées
    #  (qui existent côté externe)
    selected_in_ext = [c for c in selected_features if c in X_ext.columns]
    if len(selected_in_ext) == 0:
        selected_in_ext = selected_features  # sécurité

    proba_ext = model.predict_proba(X_ext[selected_in_ext])[:, 1]
    roc_ext, ap_ext, roc_ext_path, pr_ext_path = plot_and_save_roc_pr(
        y_ext, proba_ext, prefix="external_l1"
    )
    pd.DataFrame({"metric": ["AUROC", "AUPRC"], "value": [roc_ext, ap_ext]}).to_csv(
        os.path.join(RESULTS_DIR, "metrics_external_l1.csv"), index=False
    )
    pd.DataFrame({"y_true": y_ext.values, "y_proba": proba_ext}).to_csv(
        os.path.join(RESULTS_DIR, "predictions_external_l1.csv"), index=False
    )

    # AFFICHAGE
    print("=== Résultats enregistrés dans:", RESULTS_DIR, "===")
    print(f"[Référence/Test] AUROC = {roc_ref:.3f} | AUPRC = {ap_ref:.3f}")
    print(f"[Externe]       AUROC = {roc_ext:.3f} | AUPRC = {ap_ext:.3f}")
    print("Variables retenues (compte) :", len(selected_features))
    print("Exemples :", selected_features[: min(15, len(selected_features))])
    print("Courbes :", roc_ref_path, pr_ref_path, roc_ext_path, pr_ext_path)

if __name__ == "__main__":
    main()


