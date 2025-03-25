#!/bin/bash

# Définir les tailles de blocs à tester
BLOCK_SIZES=(1 4 8 16 32)

# Fichier de sortie pour les résultats
OUTPUT_FILE="results_cuda.txt"

# Nom de l'exécutable
EXECUTABLE="./code_cuda"

# Répertoire pour stocker les images
OUTPUT_DIR="output_images"

# Nombre d'itérations par taille de bloc (par défaut 4)
NUMBER_OF_TIME=10

# Fonction pour afficher un usage correct du script
function usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -n <number>    Nombre d'itérations (par défaut 4)"
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

# Compiler le code CUDA à chaque fois (avec nvcc)
echo "Compilation de code_cuda.cu..."
nvcc -o $EXECUTABLE code_cuda.cu -lm -O3  # Vous pouvez ajuster les options de compilation si nécessaire

# Vérifier si l'exécutable existe après la compilation
if [ ! -f "$EXECUTABLE" ]; then
    echo "Erreur : L'exécutable $EXECUTABLE n'a pas pu être créé. Vérifiez le code source."
    exit 1
fi

# Créer le répertoire pour les images de sortie si nécessaire
mkdir -p "$OUTPUT_DIR"

# Nettoyer le fichier de résultats
echo "Résultats des exécutions de code_cuda.cu" > $OUTPUT_FILE
echo "-------------------------------------" >> $OUTPUT_FILE

for BLOCK_SIZE in "${BLOCK_SIZES[@]}"; do
    total_time=0  # Variable pour stocker le temps total pour chaque taille de bloc
    output_image="${OUTPUT_DIR}/output_${BLOCK_SIZE}.bmp"  # Définir output_image avant utilisation

    # Exécuter le programme NUMBER_OF_TIME fois
    for ((i=1; i<=NUMBER_OF_TIME; i++)); do
        # Exécuter le programme et récupérer le temps d'exécution
        elapsed_time=$($EXECUTABLE lena512.bmp $output_image $BLOCK_SIZE | head -n 1)
        echo "elapsed_time: $elapsed_time"
        
        # Vérifier si le temps d'exécution a été récupéré et est un nombre valide
        if [[ -z "$elapsed_time" ]] || [[ ! "$elapsed_time" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            echo "Erreur: Temps d'exécution invalide ou non récupéré pour le bloc $BLOCK_SIZE."
            exit 1
        fi

        # Ajouter le temps d'exécution à la somme totale
        total_time=$(echo "$total_time + $elapsed_time" | bc)
    done

    # Calculer la moyenne des temps
    avg_time=$(echo "scale=6; $total_time / $NUMBER_OF_TIME" | bc)

    # Sauvegarder la moyenne dans le fichier de résultats
    echo "BLOCK_SIZE=$BLOCK_SIZE, Moyenne: $avg_time secondes" >> $OUTPUT_FILE
done


echo "Expérimentations terminées. Résultats enregistrés dans $OUTPUT_FILE."
echo "Les images pour chaque bloc ont été sauvegardées dans le répertoire $OUTPUT_DIR pour la première itération (i=1)."
