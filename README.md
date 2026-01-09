
# ChemAP (Chemical structure-based Drug Approval Prediction)

![Status](https://img.shields.io/badge/Status-Active-success)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Neural%20Networks-purple)
![Drug Prediction](https://img.shields.io/badge/Drug%20Approval-Prediction-lightcoral)
![Knowledge Distillation](https://img.shields.io/badge/Knowledge-Distillation-brown)

=============

## Aperçu général 

*ChemAP* est un cadre méthodologique conçu pour prédire la probabilité d’approbation d’un médicament à partir de sa structure chimique.  
Il repose sur une architecture enseignant–étudiant, où un modèle enseignant multimodal transmet ses connaissances à deux modèles étudiants basés uniquement sur les représentations chimiques (SMILES et ECFP4).  
Grâce à la distillation des connaissances, ChemAP associe la richesse des représentations multimodales à la simplicité et à la généralisabilité des modèles structuraux.

### Vue d’ensemble du cadre ChemAP 

![model1](img/pipelin.jpg)

<!-- ### Architecture du modèle

![model2](img/model_architecture.png) -->

## Installation et configuration

### Clonage du dépôt  

Tout d’abord, clonez le dépôt et accedez au répertoire du projet.

    git clone https://github.com/Saadem/ChemAP.git
    cd ChemAP

### Prérequis  

L’entraînement et l’évaluation de ChemAP ont été testés avec les versions suivantes :  

![Python](https://img.shields.io/badge/Python-3.9.12-blue)  
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-orange)  
![RDKit](https://img.shields.io/badge/RDKit-2022.09.5-green)  
![NumPy](https://img.shields.io/badge/NumPy-1.26.4-lightblue)  
![Pandas](https://img.shields.io/badge/Pandas-2.3.1-purple)  
![SciPy](https://img.shields.io/badge/SciPy-1.13.1-teal)  
<!---![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Active-success)--->

## Arborescence et description des repertoires et fichiers
L’arborescence ci-dessous présente la structure générale du projet **ChemAP**, incluant les principaux répertoires et scripts :
<!---
📦 ChemAP/  
 ┣ 📁 dataset/&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; # Jeux de données bruts et prétraités  
 ┣ 📁 img/                                         # Images et schémas du cadre ChemAP  
 ┣ 📁 model/                                       # Modèles entraînés (Teacher, Student, ChemBERT)  
 ┣ 📁 src/                                         # Modules Python (prétraitement, modèles, fonctions de perte, etc.)  
 ┃  ┣ 📄 Dataprocessing.py  
 ┃  ┣ 📄 loss_function.py  
 ┃  ┣ 📄 models.py  
 ┃  ┗ 📄 utils.py  
 ┣ 📄 ChemAP.py                                    # Script principal d’orchestration  
 ┣ 📄 data_preprocessing.py                        # Prétraitement et normalisation des données  
 ┣ 📄 FP_predictor_train.py                        # Entraînement du prédicteur ECFP4  
 ┣ 📄 SMILES_predictor_train.py                    # Entraînement du prédicteur SMILES  
 ┣ 📄 Teacher_train.py                             # Entraînement du modèle enseignant multimodal  
 ┣ 📄 logreg_l1.py                                 # Régression logistique pénalisée (baseline)  
 ┣ 📄 clean_log.py                                 # Nettoyage des fichiers logs  
 ┗ 📄 README.md                                    # Documentation du projet  --->

| Arborescence des repertoires et fichiers | Description des repertoires et fichiers |
|------------------------------------------|-----------------------------------------|
| ┣ 📁 `dataset/` | Jeux de données bruts et prétraités |
| ┣ 📁 `img/` | Images et schémas du cadre ChemAP |
| ┣ 📁 `model/` | Modèles entraînés (Teacher, Student, ChemBERT) |
| ┣ 📁 `src/` | Modules Python (prétraitement, modèles, fonctions de perte, etc.) |
| ┃  ┣ 📄 `Dataprocessing.py` | Préparation et transformation des jeux de données |
| ┃  ┣ 📄 `loss_function.py` | Fonctions de perte (distillation, classification, régularisation) |
| ┃  ┣ 📄 `models.py` | Architectures des modèles (enseignant, étudiants, modules auxiliaires) |
| ┃  ┣ 📄 `utils.py` | Fonctions utilitaires (chargement, métriques, visualisations, etc.) |
| ┣ 📄 `ChemAP.py` | Script principal d’orchestration du projet |
| ┣ 📄 `data_preprocessing.py` | Prétraitement et normalisation des données |
| ┣ 📄 `FP_predictor_train.py` | Entraînement du prédicteur ECFP4 |
| ┣ 📄 `SMILES_predictor_train.py` | Entraînement du prédicteur SMILES |
| ┣ 📄 `Teacher_train.py` | Entraînement du modèle enseignant multimodal |
| ┣ 📄 `logreg_l1.py` | Régression logistique pénalisée (baseline comparative) |
| ┣ 📄 `clean_log.py` | Nettoyage des fichiers logs pour éviter les erreurs d’encodage |
| ┣ 📄 `README.md` | Documentation et instructions d’utilisation du projet |


## Exemples d’utilisation  
-------------

### Préparation des jeux de données  

#### Jeu de référence 

    python data_preprocessing.py --data_path ./dataset --save_path ./dataset/processed_data --split Drug


#### Jeu de données externe  

    python data_preprocessing.py --data_path ./dataset --save_path ./dataset/processed_data --split Drug --dataset External


## Entraînement du cadre ChemAP  

*ChemAP* comprend un modèle enseignant multimodal et deux prédicteurs étudiants fondés sur la structure chimique (SMILES et ECFP4).  
Chaque prédicteur est entraîné séparément, puis la prédiction finale combine leurs probabilités d’approbation par moyenne arithmétique.

### 1. Entraînement du modèle enseignant  

    python Teacher_train.py


### 2. Entraînement du prédicteur basé sur SMILES  

Ce modèle nécessite le *modèle ChemBERT pré-entraîné*, disponible ici :  
[→ Télécharger le modèle ChemBERT](https://drive.google.com/file/d/1-8oAIwKowGy89w-ZjvCGSc1jsCWNS1Fw/view?usp=sharing)

Enregistrez le fichier téléchargé dans le répertoire suivant : `./model/ChemBERT/`


(Optionnel) 
Des versions pré-entraînées des prédicteurs peuvent également être téléchargées ici :  
[→ Modèles pré-entraînés ChemAP](https://drive.google.com/drive/folders/1hiHYnaUobdM8LiWDqrW4P0wMPDuK3hUh?usp=sharing)

- **Modèle ECFP4 :** placer dans `./model/ChemAP/ECFP_predictor/`  
- **Modèle SMILES :** placer dans `./model/ChemAP/SMILES_predictor/`  

Lancer ensuite l’entraînement :  

    python SMILES_predictor_train.py

### 3. Entraînement du prédicteur basé sur fragments 2D (ECFP4)  

    python FP_predictor_train.py


### 4. Prédiction de l’approbation avec ChemAP  

Avant cette étape, les deux prédicteurs étudiants doivent avoir été entraînés.  

    python ChemAP.py --data_type DrugApp

## Inférence  

### Jeu de données externe 

Avant l’inférence, le jeu de données externe doit être prétraité :  

    python ChemAP.py --data_type External


### Liste de médicaments utilisateur  
Pour prédire l’approbation d’une liste personnalisée de molécules :  

    python ChemAP.py --data_type custom --input_file example.csv --output example


# Régression logistique pénalisée L1 (baseline)

![Status](https://img.shields.io/badge/Status-Active-success)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Logistic%20Regression-blue)
![Drug Prediction](https://img.shields.io/badge/Drug%20Approval-Prediction-lightcoral)
![Baseline](https://img.shields.io/badge/Baseline-L1%20Regularization-orange)
![Author](https://img.shields.io/badge/Author-Thierry_F_Saadem-purple)
![License](https://img.shields.io/badge/License-Ulaval-yellow)

Le modèle de régression logistique pénalisée L1 sert de baseline comparative pour la prédiction d’approbation des médicaments.  

lancer l'entrainement :

    python logreg_l1.py --data_path ./dataset/processed_data --split Drug --output_path ./model/logreg_l1


# Référence  

![Reference](https://img.shields.io/badge/Reference-ChemAP-green)
![Source Code](https://img.shields.io/badge/Source-GitHub-blue)
![License](https://img.shields.io/badge/License-MIT-red)
![Author](https://img.shields.io/badge/Author-Thierry_F_Saadem-purple)


Les codes et scripts du cadre *ChemAP* présentés dans ce dépôt ont été inspirés et partiellement adaptés du travail original de Changyun Cho, Sangseon Lee, Dongmin Bang, Yinhua Piao & Sun Kim (2024), dont les sources sont disponibles [içi](https://github.com/ChangyunCho/ChemAP)    
