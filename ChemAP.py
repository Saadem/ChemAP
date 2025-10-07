import os
import argparse
import csv
import pickle
import numpy as np 
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.swa_utils import AveragedModel
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.models import Multimodal_Teacher, FP_Student, SMILES_BERT, SMILES_Student
from src.Dataprocessing import Dataset, External_Dataset
from src.loss_function import DistillationLoss
from src.utils import *

from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', help="Data type", type=str, default='DrugApp')
    parser.add_argument('--data_path', help="processed dataset path", type=str, default='./dataset/processed_data')
    parser.add_argument('--input_file', help="user file", type=str, default='example.csv')
    parser.add_argument('--output', help="output path", type=str, default='example')
    parser.add_argument('--model_path', help="trained model path", type=str, default='./model/ChemAP')
    parser.add_argument('--fp_dim_1', help='2D fragment predictor hidden dim 1', type=int, default=1024)
    parser.add_argument('--fp_dim_2', help='2D fragment predictor hidden dim 2', type=int, default=128)
    parser.add_argument('--fp_dim_3', help='2D fragment predictor hidden dim 3', type=int, default=256)
    parser.add_argument('--fp_drop_1', help='2D fragment predictor dropout rate 1', type=float, default=0.21)
    parser.add_argument('--fp_drop_2', help='2D fragment predictor dropout rate 2', type=float, default=0.11)
    parser.add_argument("--KD", help="Knowledge distillation", default=True)
    parser.add_argument('--gpu', help="gpu device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    arg = parser.parse_args()
    
    # Info CUDA
    if not torch.cuda.is_available():
        print("⚠️ CUDA non disponible — initialisation en mode CPU.")

    print("Initialisation du dispositif (CPU ou GPU)...")
    device = torch.device(f"cuda:{arg.gpu}" if torch.cuda.is_available() else "cpu")
    
    if arg.data_type == 'DrugApp':
        print('📦 Prédiction d’approbation de médicament sur DrugApp')
    elif arg.data_type == 'External':
        print('📦 Prédiction d’approbation sur un jeu de données externe')
    elif arg.data_type == 'custom':
        print('📦 Prédiction d’approbation sur un jeu personnalisé')
    else:
        print('❌ Type de données non reconnu')

    model_saved_dir = './model/ChemAP'
    perform_save_dir = f'./results'
    
    if arg.KD == False:
        KD = '_wo_KD'
    elif arg.KD == True:
        KD = ''

    os.makedirs(perform_save_dir, exist_ok=True)
    print(f"📁 Répertoire de résultats créé : {perform_save_dir}")

    if arg.data_type == 'External':
        print("📝 Initialisation du fichier CSV pour résultats externes...")
        f = open(f'{perform_save_dir}/External_pred_statistics.csv', 'w', newline='')
        wr = csv.writer(f)
        wr.writerow(['model seed', 'External dataset drug number', 'ChemAP pred'])
        f.close()

    # FP model arguments (gardés comme dans le code de base)
    fp_enc_h1 = 1024
    fp_enc_h2 = 128
    fp_enc_d = 0.21
    fp_pro_h1 = 256
    fp_pro_d = 0.11

    roc_ls_t = []
    prc_ls_t = []
    acc_ls_t = []
    pre_ls_t = []
    rec_ls_t = []
    f1_ls_t  = []
    ba_ls_t  = []

    temp = []

    print("🔁 Initialisation aléatoire et vocabulaire SMILES...")
    seed_everything(arg.seed)
    Smiles_vocab = Vocab()

    print("📄 Chargement des données de test...")
    if arg.data_type == 'DrugApp':
        test = pd.read_csv(f'{arg.data_path}/test/DrugApp_seed_{arg.seed}_test_minmax.csv')
        test_dataset = Dataset(test, device, model_type='ChemAP', vocab=Smiles_vocab, seq_len=256)

    elif arg.data_type == 'External':
        train = pd.read_csv(f'{arg.data_path}/train/DrugApp_seed_{arg.seed}_train_minmax.csv')
        df = pd.read_csv(f'{arg.data_path}/External/External.csv').dropna().reset_index(drop=True)
        # (Optionnel) un petit log utile
        # print("Colonnes External:", df.columns.tolist())
        test_dataset = External_Dataset(Smiles_vocab, df, 'External', device, trainset=train, similarity_cut=0.7)
        
    elif arg.data_type == 'custom':
        df = pd.read_csv(f'./dataset/{arg.input_file}')
        test_dataset = External_Dataset(Smiles_vocab, df, 'custom', device, trainset=None)

    print("📦 Création du DataLoader de test...")
    test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False)

    print("📥 Chargement des modèles entraînés (ECFP et SMILES)...")
    ecfp_student = FP_Student(2048, arg.fp_dim_1, arg.fp_dim_2, arg.fp_dim_3, arg.fp_drop_1, arg.fp_drop_2).to(device)
    ecfp_student.load_state_dict(torch.load(f'{arg.model_path}/ECFP_predictor{KD}/ECFP_predictor_{arg.seed}.pt', map_location=device))

    smiles_encoder = SMILES_BERT(len(Smiles_vocab), 
                                 max_len=256, 
                                 nhead=16, 
                                 feature_dim=1024, 
                                 feedforward_dim=1024, 
                                 nlayers=8, 
                                 adj=True,
                                 dropout_rate=0)
    smiles_student = SMILES_Student(smiles_encoder, 1024).to(device)
    smiles_student.load_state_dict(torch.load(f'{arg.model_path}/SMILES_predictor{KD}/SMILES_predictor_{arg.seed}.pt', map_location=device))

    print("🔍 Début de l'inférence...")
    ecfp_pred = []
    ecfp_prob = []
    smi_pred = []
    smi_prob = []
    target_list = []

    ecfp_student.eval()
    smiles_student.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            ecfp_2048, smi_bert_input, smi_bert_adj, smi_bert_adj_mask, y = data
            position_num = torch.arange(256).repeat(smi_bert_input.size(0),1).to(device)

            _, ecfp_output = ecfp_student(ecfp_2048)
            pred = torch.argmax(F.softmax(ecfp_output, dim=1), dim=1).detach().cpu()
            prob = F.softmax(ecfp_output, dim=1)[:,1].detach().cpu()
            ecfp_pred.append(pred)
            ecfp_prob.append(prob)

            _, smiles_output = smiles_student(smi_bert_input,
                                              position_num,
                                              smi_bert_adj_mask,
                                              smi_bert_adj)
            pred = torch.argmax(F.softmax(smiles_output, dim=1), dim=1).detach().cpu()
            prob = F.softmax(smiles_output, dim=1)[:,1].detach().cpu()
            smi_pred.append(pred)
            smi_prob.append(prob)

            target_list.append(y.cpu())

    print("✅ Fusion des prédictions ECFP + SMILES...")
    target_list = torch.cat(target_list, dim=0).numpy()
    ecfp_pred = torch.cat(ecfp_pred, dim=0).numpy()
    ecfp_prob = torch.cat(ecfp_prob, dim=0).numpy()
    smi_pred  = torch.cat(smi_pred, dim=0).numpy()
    smi_prob  = torch.cat(smi_prob, dim=0).numpy()

    ens_prob  = (ecfp_prob + smi_prob)/2
    ens_pred  = (ens_prob > 0.5)*1

    # ================== CORRECTIF LABELS EXTERNAL/CUSTOM (évaluation + graphes) ==================
    def _coerce_labels(arr_like):
        """Convertit labels en numpy array {0,1}, gère strings et NaN."""
        if arr_like is None:
            return None
        s = pd.Series(arr_like).copy()
        s = pd.to_numeric(s, errors='coerce')  # '0'/'1' -> 0/1
        s = s.dropna().astype(int).clip(lower=0, upper=1)
        return s.to_numpy()

    if arg.data_type in ['External', 'custom']:
        ext_labels = None

        # 1) Essayer les labels sortis du DataLoader (target_list)
        try:
            tl = _coerce_labels(target_list)
            if tl is not None and len(tl) == len(ens_pred) and set(np.unique(tl)) == {0, 1}:
                ext_labels = tl
                print("ℹ️ Labels détectés via DataLoader.")
        except Exception:
            pass

        # 2) Sinon, récupérer depuis la DataFrame finale du dataset
        if ext_labels is None:
            try:
                df_used = test_dataset.GetDataset()  # ordre aligné avec le DataLoader
                for col in ['Approval', 'label', 'y', 'Label']:
                    if col in df_used.columns:
                        cand = _coerce_labels(df_used[col].values)
                        if cand is not None and len(cand) == len(ens_pred) and set(np.unique(cand)) == {0, 1}:
                            ext_labels = cand
                            print(f"ℹ️ Labels détectés depuis GetDataset() colonne '{col}'.")
                            break
            except Exception:
                pass

        # 3) Si labels valides, calcul des métriques + graphes
        if ext_labels is not None:
            print("📊 Évaluation External/custom (AUROC/AUPRC)…")
            # ROC/PR
            fpr_ens, tpr_ens, _ = metrics.roc_curve(ext_labels, ens_prob, pos_label=1)
            auc_ens = metrics.auc(fpr_ens, tpr_ens)
            precision_ens, recall_ens, _ = metrics.precision_recall_curve(ext_labels, ens_prob, pos_label=1)
            AUPRC_ens = metrics.auc(recall_ens, precision_ens)
            ap_ens = metrics.average_precision_score(ext_labels, ens_prob)

            # métriques
            acc = metrics.accuracy_score(ext_labels, ens_pred)
            pre = metrics.precision_score(ext_labels, ens_pred, pos_label=1, zero_division=0)
            rec = metrics.recall_score(ext_labels, ens_pred, pos_label=1, zero_division=0)
            f1  = metrics.f1_score(ext_labels, ens_pred, pos_label=1, zero_division=0)
            ba  = metrics.balanced_accuracy_score(ext_labels, ens_pred)
            print(f"F1-score student (External/custom) : {f1:.4f}")

            # Sauvegardes
            pd.DataFrame([{
                "AUROC": auc_ens, "AUPRC": AUPRC_ens, "AP": ap_ens,
                "ACC": acc, "PRE": pre, "REC": rec, "F1": f1, "BA": ba
            }]).to_csv(f"{perform_save_dir}/External_metrics.csv", index=False)

            pd.DataFrame({"fpr": fpr_ens, "tpr": tpr_ens}).to_csv(f"{perform_save_dir}/External_roc_points.csv", index=False)
            pd.DataFrame({"recall": recall_ens, "precision": precision_ens}).to_csv(f"{perform_save_dir}/External_pr_points.csv", index=False)

            # Graphes
            plt.figure(figsize=(7,6))
            plt.plot(fpr_ens, tpr_ens, label=f'ChemAP (AUROC = {auc_ens:.3f})', linewidth=2.0)
            plt.plot([0,1],[0,1],'k--',linewidth=1)
            plt.xlim(0,1); plt.ylim(0,1)
            plt.xlabel('Taux de faux positifs (FPR)'); plt.ylabel('Taux de vrais positifs (TPR)')
            plt.legend(loc='lower right'); plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout(); plt.savefig(f'{perform_save_dir}/roc_external.png', dpi=300); plt.close()

            pos_rate = np.mean(ext_labels)
            plt.figure(figsize=(7,6))
            plt.plot(recall_ens, precision_ens, label=f'ChemAP (AP={ap_ens:.3f}, AUPRC={AUPRC_ens:.3f})', linewidth=2.0)
            plt.hlines(pos_rate, 0, 1, linestyles='--', linewidth=1)  # baseline
            plt.xlim(0,1); plt.ylim(0,1)
            plt.xlabel('Rappel'); plt.ylabel('Précision')
            plt.legend(loc='lower left'); plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout(); plt.savefig(f'{perform_save_dir}/pr_external.png', dpi=300); plt.close()

            print(f"✅ External: AUROC={auc_ens:.3f}, AUPRC={AUPRC_ens:.3f}, AP={ap_ens:.3f}, ACC={acc:.3f}, F1={f1:.3f}")
        else:
            print("ℹ️ Aucun label binaire 0/1 valide détecté : pas de ROC/PR.")
    # ================== FIN CORRECTIF ==================

    if arg.data_type == 'DrugApp':
        print("📊 Évaluation sur DrugApp...")
        fpr, tpr, thresholds = metrics.roc_curve(target_list, ens_prob, pos_label=1)
        roc_ls_t.append(metrics.auc(fpr, tpr))
        precision, recall, _ = metrics.precision_recall_curve(target_list, ens_prob, pos_label=1)
        prc_ls_t.append(metrics.auc(recall, precision))
        acc_ls_t.append(metrics.accuracy_score(target_list, ens_pred))
        pre_ls_t.append(metrics.precision_score(target_list, ens_pred, pos_label=1))
        rec_ls_t.append(metrics.recall_score(target_list, ens_pred, pos_label=1))
        f1_ls_t.append(metrics.f1_score(target_list, ens_pred, pos_label=1))
        ba_ls_t.append(metrics.balanced_accuracy_score(target_list, ens_pred))

        roc_t = pd.DataFrame(roc_ls_t, columns = ['AUCROC'])
        prc_t = pd.DataFrame(prc_ls_t, columns = ['AUPRC'])
        acc_t = pd.DataFrame(acc_ls_t, columns = ['ACC'])
        pre_t = pd.DataFrame(pre_ls_t, columns = ['PRE'])
        rec_t = pd.DataFrame(rec_ls_t, columns = ['REC'])
        f1_t  = pd.DataFrame(f1_ls_t, columns = ['F1'])
        ba_t  = pd.DataFrame(ba_ls_t, columns = ['BA'])

        res_t = pd.concat([roc_t, prc_t, acc_t, ba_t, f1_t, pre_t, rec_t], axis=1)
        res_t.to_csv(f'{perform_save_dir}/ChemAP_DrugApp_test_perform.csv', sep = ',', index=None)

        print('📁 Performance du modèle ChemAP enregistrée pour DrugApp.')

        # === Bloc courbes ROC ===
        fpr_ecfp, tpr_ecfp, _ = roc_curve(target_list, ecfp_prob, pos_label=1)
        auc_ecfp = auc(fpr_ecfp, tpr_ecfp)

        fpr_smi, tpr_smi, _ = roc_curve(target_list, smi_prob, pos_label=1)
        auc_smi = auc(fpr_smi, tpr_smi)

        fpr_ens, tpr_ens, _ = roc_curve(target_list, ens_prob, pos_label=1)
        auc_ens = auc(fpr_ens, tpr_ens)

        print(f"AUROC ECFP   : {auc_ecfp:.4f}")
        print(f"AUROC SMILES : {auc_smi:.4f}")
        print(f"AUROC student (moyenne) : {auc_ens:.4f}")
        print(f"F1-score student (DrugApp) : {f1_ls_t[-1]:.4f}")


        # Sauvegarde fpr/tpr ensemble
        df_ens = pd.DataFrame({
            "fpr_ens": fpr_ens,
            "tpr_ens": tpr_ens
        })
        df_ens.to_csv("./results/roc_ens.csv", index=False)
        print("✅ fpr_ens et tpr_ens enregistrés dans ./results/roc_ens.csv")

        plt.figure(figsize=(7, 6))
        plt.plot(fpr_ecfp, tpr_ecfp, label=f'ECFP (AUROC = {auc_ecfp:.3f})')
        plt.plot(fpr_smi, tpr_smi, label=f'SMILES (AUROC = {auc_smi:.3f})')
        plt.plot(fpr_ens, tpr_ens, label=f'student (AUROC = {auc_ens:.3f})', linewidth=2.2)

        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)  # Diagonale aléatoire

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('Taux de faux positifs (FPR)')
        plt.ylabel('Taux de vrais positifs (TPR)')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig('./results/roc_student.png', dpi=300)
        plt.show()

        # === Courbe Précision–Rappel (et aires) ===
        pre_ecfp, rec_ecfp, _ = metrics.precision_recall_curve(target_list, ecfp_prob, pos_label=1)
        ap_ecfp  = metrics.average_precision_score(target_list, ecfp_prob)
        AUPRC_ecfp = metrics.auc(rec_ecfp, pre_ecfp)

        pre_smi, rec_smi, _ = metrics.precision_recall_curve(target_list, smi_prob, pos_label=1)
        ap_smi  = metrics.average_precision_score(target_list, smi_prob)
        AUPRC_smi = metrics.auc(rec_smi, pre_smi)

        pre_ens, rec_ens, _ = metrics.precision_recall_curve(target_list, ens_prob, pos_label=1)
        ap_ens  = metrics.average_precision_score(target_list, ens_prob)
        AUPRC_ens = metrics.auc(rec_ens, pre_ens)

        # (optionnel) sauvegarder la courbe PR de l'ensemble
        pd.DataFrame({"recall_ens": rec_ens, "precision_ens": pre_ens}).to_csv("./results/pr_ens.csv", index=False)

        # Prévalence (ligne de base PR)
        pos_rate = np.mean(target_list)

        plt.figure(figsize=(7, 6))
        plt.plot(rec_ecfp, pre_ecfp, label=f'ECFP (AP={ap_ecfp:.3f}, AUPRC={AUPRC_ecfp:.3f})')
        plt.plot(rec_smi,  pre_smi,  label=f'SMILES (AP={ap_smi:.3f}, AUPRC={AUPRC_smi:.3f})')
        plt.plot(rec_ens,  pre_ens,  label=f'student (AP={ap_ens:.3f}, AUPRC={AUPRC_ens:.3f})', linewidth=2.2)
        plt.hlines(pos_rate, 0, 1, linestyles='--', linewidth=1)  # baseline = prévalence

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('Rappel')
        plt.ylabel('Précision')
        plt.legend(loc='lower left')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig('./results/pr_student.png', dpi=300)
        plt.show()



    elif arg.data_type == 'External':
        print('📁 Sauvegarde des prédictions pour le jeu de données externe...')
        f = open(f'{perform_save_dir}/External_pred_statistics.csv', 'a', newline='')
        wr = csv.writer(f)
        wr.writerow([arg.seed, test_dataset.__len__(), sum(ens_pred)])
        f.close()
        dataset = test_dataset.GetDataset()
        dataset['ChemAP_pred'] = ens_pred
        dataset.to_csv(f'{perform_save_dir}/External_prediction.csv', sep=',', index=None)
        print('✅ Prédictions ChemAP sauvegardées (externe).')

    elif arg.data_type == 'custom':
        print('📁 Sauvegarde des prédictions pour le jeu personnalisé...')
        dataset = test_dataset.GetDataset()
        dataset['ChemAP_pred'] = ens_pred
        dataset.to_csv(f'{perform_save_dir}/{arg.output}_prediction.csv', sep=',', index=None)
        print('✅ Prédictions ChemAP sauvegardées (personnalisé).')

if __name__ == "__main__":
    main()
