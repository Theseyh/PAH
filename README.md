
# Filtrage Bilatéral avec CUDA

## 1. Introduction

Le filtrage bilatéral est une technique avancée de traitement d'image qui permet de réduire le bruit tout en préservant les contours et les détails. Contrairement aux filtres de lissage classiques comme le flou gaussien, qui ont tendance à estomper les bords, le filtre bilatéral applique un lissage spatial pondéré en fonction de la différence d'intensité entre les pixels. Cela le rend particulièrement utile en vision par ordinateur, en imagerie médicale et en traitement HDR.

## 2. Contexte et problématique

Les filtres classiques de lissage, bien qu'efficaces pour réduire le bruit, ont pour inconvénient d'altérer les contours et les détails fins des images. Le filtre bilatéral résout ce problème en prenant en compte non seulement la distance spatiale entre les pixels, mais aussi la différence d'intensité entre eux. Cette approche permet de préserver les bords tout en réduisant le bruit de l'image.

## 3. Formulation mathématique

Le filtre bilatéral est défini comme suit :

voir formule

Avec :

-   I(x): Intensité initiale du pixel x.
    
-   I′(x): Intensité filtrée du pixel x.
    
-   W(x) : Facteur de normalisation, défini par :
    
voir formule
    
-   σs : Paramètre de lissage spatial (contrôle l'influence des pixels en fonction de leur distance).
    
-   σr: Paramètre de préservation des contours (contrôle la sensibilité aux variations d'intensité).
    
-   d : Fenêtre de filtrage, définissant les pixels pris en compte autour de x.
    

## 4. Implémentation

### 4.1 Structure des fichiers

-   `code.c` : Implémentation séquentielle du filtre bilatéral en C.
    
-   `code.cu` : Implémentation parallèle en CUDA.
    
-   `lena_grey.bmp` : Image d'entrée en niveaux de gris utilisée pour le traitement.
    

### 4.2 Implémentation séquentielle en C

L'implémentation séquentielle consiste à :

-   Charger et manipuler une image.
    
-   Appliquer le filtre bilatéral sans utiliser de bibliothèques tierces.
    
-   Sauvegarder l'image filtrée.
    
-   Vérifier le bon fonctionnement sur une image bruitée.
    

### 4.3 Portage vers CUDA

Le passage à CUDA pose plusieurs défis, notamment :

-   La gestion des accès mémoire et de la fenêtre de voisinage.
    
-   L'organisation des threads et des blocs.
    
-   L'utilisation efficace de la mémoire partagée.
    

L'implémentation CUDA optimise le traitement en parallélisant le calcul pour chaque pixel.

## 5. Analyse des performances

Une comparaison des performances a été réalisée entre l'implémentation séquentielle en C et l'implémentation CUDA.

Implémentation

Temps d'exécution (s)

C Séquentiel

0.44

CUDA

0.22

Le code CUDA est donc **deux fois plus rapide** que l'implémentation séquentielle.

## 6. Améliorations possibles

Bien que l'implémentation CUDA soit déjà optimisée, plusieurs améliorations peuvent encore être envisagées :

-   **Utilisation accrue de la mémoire partagée** : Réduire les accès à la mémoire globale en stockant localement les valeurs dans chaque bloc.
    
-   **Optimisation des boucles** : Paralléliser davantage les calculs internes pour réduire la charge de travail de chaque thread.
    
-   **Utilisation de textures CUDA** : Exploiter la mémoire texture pour des accès plus rapides aux données.
    
-   **Optimisation des tailles de blocs** : Tester différentes tailles de blocs pour maximiser l'occupation des multiprocesseurs.
    

Ces améliorations pourraient permettre d'améliorer encore les performances et de réduire le temps de calcul global du filtre bilatéral en CUDA.
