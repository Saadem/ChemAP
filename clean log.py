import pathlib

# Chemin vers le fichier logreg_l1.py
fichier = pathlib.Path(__file__).parent / "logreg_l1.py"

# Lecture et nettoyage
texte = fichier.read_text(encoding="utf-8")
texte_propre = texte.replace("\u00A0", " ")

# Réécriture du fichier
fichier.write_text(texte_propre, encoding="utf-8")

print("✅ Espaces insécables remplacés dans :", fichier)
