# Importation des bibliothèques nécessaires
import os  # pour la gestion des chemins et répertoires
import argparse  # pour la gestion des arguments passés en ligne de commande
import pandas as pd  # pour la manipulation de données tabulaires

# Importation de fonctions personnalisées à partir du projet local
from src.utils import seed_everything  # pour fixer la graine aléatoire
from src.Dataprocessing import DatasetSplit  # classe pour le découpage du jeu de données

from sklearn.preprocessing import MinMaxScaler  # pour normaliser les caractéristiques

# Fonction utilitaire pour calculer la longueur de la chaîne SMILES d'une molécule
def smiles_length(df):
    return len(df['SMILES'])

# Fonction principale
def main():
    # Définition des arguments utilisables en ligne de commande
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help="dataset path", type=str, default = './dataset')
    parser.add_argument('--save_path', help="processed data save path", type=str, default = './dataset/processed_data')
    parser.add_argument('--dataset', help="DrugApp or External", type=str, default= 'DrugApp')
    parser.add_argument("--split", help="data split type", type=str, default='Drug')
    parser.add_argument("--seed", type=int, default=7)
    arg = parser.parse_args()
    
    # Traitement du jeu de données principal (DrugApp)
    if arg.dataset == 'DrugApp':
        print('Benchmark dataset processing')

        # Création des dossiers de sauvegarde pour les jeux de données traités
        os.makedirs(f'{arg.save_path}/train', exist_ok=True)
        os.makedirs(f'{arg.save_path}/valid', exist_ok=True)
        os.makedirs(f'{arg.save_path}/test', exist_ok=True)

        # Lecture du fichier CSV contenant l’ensemble des vecteurs de caractéristiques
        total_dataset = pd.read_csv(f'{arg.data_path}/DrugApp/All_training_feature_vectors.csv')

        # Ajout d’une colonne de longueurs SMILES pour filtrage
        total_dataset['length'] = total_dataset.apply(smiles_length, axis=1)
        
        # Suppression des molécules ayant une chaîne SMILES de plus de 256 caractères
        total_dataset = total_dataset[total_dataset['length'] <= 256].drop(columns='length').reset_index(drop=True)

        # Initialisation de la graine aléatoire pour la reproductibilité
        seed_everything(arg.seed)

        # Instanciation d’un objet de découpage (selon le type précisé) et découpage en train, valid, test
        dataset = DatasetSplit(total_dataset, split=arg.split)
        train, valid, test = dataset.data_split()

        # Sauvegarde des jeux sans normalisation (utile pour modèles classiques de ML)
        train.to_csv(f'{arg.save_path}/train/DrugApp_seed_{arg.seed}_train_no_scaler.csv', sep=',', index=None)
        valid.to_csv(f'{arg.save_path}/valid/DrugApp_seed_{arg.seed}_valid_no_scaler.csv', sep=',', index=None)
        test.to_csv(f'{arg.save_path}/test/DrugApp_seed_{arg.seed}_test_no_scaler.csv', sep=',', index=None)

        # Initialisation du normaliseur MinMax et ajustement sur les variables explicatives (colonnes 2 à 57)
        scaler = MinMaxScaler()
        scaler.fit(train.iloc[:,2:58])  # normalisation sur les features uniquement

        # Application du scaler aux données d'entraînement, validation et test
        train = pd.concat([train.iloc[:,:2], pd.DataFrame(scaler.transform(train.iloc[:,2:58]))],axis=1)
        valid = pd.concat([valid.iloc[:,:2], pd.DataFrame(scaler.transform(valid.iloc[:,2:58]))],axis=1)
        test = pd.concat([test.iloc[:,:2], pd.DataFrame(scaler.transform(test.iloc[:,2:58]))],axis=1)

        # Sauvegarde des données normalisées (utile pour modèles de deep learning)
        train.to_csv(f'{arg.save_path}/train/DrugApp_seed_{arg.seed}_train_minmax.csv', sep=',', index=None)
        valid.to_csv(f'{arg.save_path}/valid/DrugApp_seed_{arg.seed}_valid_minmax.csv', sep=',', index=None)
        test.to_csv(f'{arg.save_path}/test/DrugApp_seed_{arg.seed}_test_minmax.csv', sep=',', index=None)

        print(f'Data processing with {arg.split} split is done')
    
    # Traitement d’un jeu de données externe (FDA 2023 + Clinical Trials 2024)
    elif arg.dataset == 'External':
        print('FDA Approved 2023 and ClinicalTrials Failed 2024 dataset processing')

        # Création du répertoire de sauvegarde pour ce jeu externe
        os.makedirs(f'{arg.save_path}/External', exist_ok=True)

        # Lecture et sélection des colonnes pertinentes dans les deux jeux
        fda = pd.read_csv('./dataset/FDA/FDA_2023_approved.csv')[['Drug Name', 'SMILES']].dropna().reset_index(drop=True)
        clinical = pd.read_csv('./dataset/ClinicalTrials/clinical_fail_2024_05.csv')[['Name', 'SMILES']].dropna().reset_index(drop=True)

        # Ajout des étiquettes d’approbation : 1 pour FDA, 0 pour ClinicalTrials
        fda['Approval'] = 1
        clinical['Approval'] = 0

        # Harmonisation du nom de colonne pour concaténation
        clinical.columns = ['Drug Name', 'SMILES', 'Approval']

        # Fusion des deux jeux en un seul
        external = pd.concat([fda, clinical], axis=0).reset_index(drop=True)

        # Sauvegarde du jeu de données externe combiné
        external.to_csv(f'{arg.save_path}/External/External.csv', sep=',', index=None)

# Point d’entrée du script
if __name__ == "__main__":
    main()





# import os
# import argparse
# import pandas as pd

# from src.utils import seed_everything
# from src.Dataprocessing import DatasetSplit

# from sklearn.preprocessing import MinMaxScaler

# def smiles_length(df):
#     return len(df['SMILES'])

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_path', help="dataset path", type=str, default = './dataset')
#     parser.add_argument('--save_path', help="processed data save path", type=str, default = './dataset/processed_data')
#     parser.add_argument('--dataset', help="DrugApp or External", type=str, default= 'DrugApp')
#     parser.add_argument("--split", help="data split type", type=str, default='Drug')
#     parser.add_argument("--seed", type=int, default=7)
#     arg = parser.parse_args()
    
#     if arg.dataset == 'DrugApp':
#         print('Benchmark dataset processing')
#         os.makedirs(f'{arg.save_path}/train', exist_ok=True)
#         os.makedirs(f'{arg.save_path}/valid', exist_ok=True)
#         os.makedirs(f'{arg.save_path}/test', exist_ok=True)

#         total_dataset = pd.read_csv(f'{arg.data_path}/DrugApp/All_training_feature_vectors.csv')

#         # remove SMILES length over 256
#         total_dataset['length'] = total_dataset.apply(smiles_length, axis=1)
#         total_dataset[total_dataset['length'] <= 256].drop(columns='length').reset_index(drop=True)

#         # Dataset split (train, valid, test)    
#         seed_everything(arg.seed)
#         dataset = DatasetSplit(total_dataset, split=arg.split)
#         train, valid, test = dataset.data_split()

#         # without scaler for training ML models
#         train.to_csv(f'{arg.save_path}/train/DrugApp_seed_{arg.seed}_train_no_scaler.csv', sep=',', index=None)
#         valid.to_csv(f'{arg.save_path}/valid/DrugApp_seed_{arg.seed}_valid_no_scaler.csv', sep=',', index=None)
#         test.to_csv(f'{arg.save_path}/test/DrugApp_seed_{arg.seed}_test_no_scaler.csv', sep=',', index=None)

#         # feature scaling for training DL models   
#         scaler = MinMaxScaler()
#         scaler.fit(train.iloc[:,2:58])
#         train = pd.concat([train.iloc[:,:2], pd.DataFrame(scaler.transform(train.iloc[:,2:58]))],axis=1)
#         valid = pd.concat([valid.iloc[:,:2], pd.DataFrame(scaler.transform(valid.iloc[:,2:58]))],axis=1)
#         test = pd.concat([test.iloc[:,:2], pd.DataFrame(scaler.transform(test.iloc[:,2:58]))],axis=1)

#         train.to_csv(f'{arg.save_path}/train/DrugApp_seed_{arg.seed}_train_minmax.csv', sep=',', index=None)
#         valid.to_csv(f'{arg.save_path}/valid/DrugApp_seed_{arg.seed}_valid_minmax.csv', sep=',', index=None)
#         test.to_csv(f'{arg.save_path}/test/DrugApp_seed_{arg.seed}_test_minmax.csv', sep=',', index=None)

#         print(f'Data processing with {arg.split} split is done')
    
#     elif arg.dataset == 'External':
#         print('FDA Approved 2023 and ClinicalTrials Failed 2024 dataset processing')
#         os.makedirs(f'{arg.save_path}/External', exist_ok=True)
#         fda = pd.read_csv('./dataset/FDA/FDA_2023_approved.csv')[['Drug Name', 'SMILES']].dropna().reset_index(drop=True)
#         clinical = pd.read_csv('./dataset/ClinicalTrials/clinical_fail_2024_05.csv')[['Name', 'SMILES']].dropna().reset_index(drop=True)
#         fda['Approval'] = 1
#         clinical['Approval'] = 0
#         clinical.columns = ['Drug Name', 'SMILES', 'Approval']
#         external = pd.concat([fda, clinical], axis=0).reset_index(drop=True)
#         external.to_csv(f'{arg.save_path}/External/External.csv', sep=',', index=None)

# if __name__ == "__main__":
#     main()