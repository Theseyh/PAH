#!/bin/bash

# Fichier de sortie pour les résultats
OUTPUT_FILE="result.txt"

# Nom de l'exécutable
EXECUTABLE="./code"

# Répertoire pour stocker les images
OUTPUT_DIR="output_images"

# Nombre d'itérations (par défaut 10)
NUMBER_OF_TIME=10

# Fonction pour afficher un usage correct du script
function usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -n <number>    Nombre d'itérations (par défaut 10)"
    echo "  -h             Afficher ce message d'aide"
    exit 1
}

# Analyser les arguments
while getopts "n:h" opt; do
    case "$opt" in
        n) NUMBER_OF_TIME=$OPTARG ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Compiler le code C à chaque fois
echo "Compilation de code.c..."
gcc -o $EXECUTABLE code.c -lm -O3  # Vous pouvez ajuster les options de compilation si nécessaire

# Vérifier si l'exécutable existe après la compilation
if [ ! -f "$EXECUTABLE" ]; then
    echo "Erreur : L'exécutable $EXECUTABLE n'a pas pu être créé. Vérifiez le code source."
    exit 1
fi

# Créer le répertoire pour les images de sortie si nécessaire
mkdir -p "$OUTPUT_DIR"

# Nettoyer le fichier de résultats
echo "Résultats des exécutions de code.c" > $OUTPUT_FILE
echo "-------------------------------------" >> $OUTPUT_FILE

# Exécuter le programme pour NUMBER_OF_TIME itérations
total_time=0  # Variable pour stocker le temps total

for ((i=1; i<=NUMBER_OF_TIME; i++)); do
    # Définir le nom de l'image de sortie
    output_image="${OUTPUT_DIR}/output.bmp"

    # Exécuter le programme et récupérer le temps d'exécution
    elapsed_time=$($EXECUTABLE lena512.bmp $output_image | head -n 1)
    echo "elapsed_time: $elapsed_time"
    # Vérifier si le temps d'exécution a été récupéré et est un nombre valide
    if [[ -z "$elapsed_time" ]] || [[ ! "$elapsed_time" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        echo "Erreur: Temps d'exécution invalide ou non récupéré pour l'itération $i."
        echo "elapsed_time: $elapsed_time"
        exit 1
    fi

    # Ajouter le temps d'exécution à la somme totale
    total_time=$(echo "$total_time + $elapsed_time" | bc)
done

# Calculer la moyenne des temps
avg_time=$(echo "scale=6; $total_time / $NUMBER_OF_TIME" | bc)

# Sauvegarder la moyenne dans le fichier de résultats
echo "Moyenne des temps pour $NUMBER_OF_TIME itérations: $avg_time secondes" >> $OUTPUT_FILE

echo "Expérimentations terminées. Résultats enregistrés dans $OUTPUT_FILE."
echo "Les images pour chaque itération ont été sauvegardées dans le répertoire $OUTPUT_DIR."
