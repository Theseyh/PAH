
# Filtrage Bilatéral avec CUDA

## 1. Introduction

Le filtrage bilatéral est une technique avancée de traitement d'image qui permet de réduire le bruit tout en préservant les contours et les détails. Contrairement aux filtres de lissage classiques comme le flou gaussien, qui ont tendance à estomper les bords, le filtre bilatéral applique un lissage spatial pondéré en fonction de la différence d'intensité entre les pixels. Cela le rend particulièrement utile en vision par ordinateur, en imagerie médicale et en traitement HDR.

## 2. Contexte et problématique

Les filtres classiques de lissage, bien qu'efficaces pour réduire le bruit, ont pour inconvénient d'altérer les contours et les détails fins des images. Le filtre bilatéral résout ce problème en prenant en compte non seulement la distance spatiale entre les pixels, mais aussi la différence d'intensité entre eux. Cette approche permet de préserver les bords tout en réduisant le bruit de l'image.

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

Le portage du code sur CUDA implique plusieurs défis techniques, notamment :

- **La gestion des accès mémoire** : En CUDA, l'accès à la mémoire doit être optimisé pour éviter les goulots d'étranglement. Il est essentiel de gérer la mémoire globale, partagée et constante de manière efficace.
  
- **L'organisation des threads et des blocs** : Les threads doivent être organisés en blocs pour exploiter le parallélisme sur le GPU. La bonne configuration de la taille des blocs et de la grille est cruciale pour de bonnes performances.
  
- **L'utilisation efficace de la mémoire partagée** : En CUDA, la mémoire partagée est un espace mémoire rapide utilisé par tous les threads d'un même bloc. Elle doit être utilisée judicieusement pour éviter des accès inutiles à la mémoire globale, qui est plus lente.

#### Stratégie de parallélisation

Dans cette application, l'objectif est de paralléliser le calcul d'un filtre bilatéral. Pour ce faire, deux parties principales du code sont parallélisées :

1. **Le calcul des poids spatiaux (`spatial_weights`)** : Le noyau spatial est une matrice carrée de taille `d x d` qui représente les poids de la fenêtre pour chaque pixel. Ces poids dépendent de la distance entre le pixel central et ses voisins.

2. **L'application du filtre bilatéral à l'image** : Pour chaque pixel de l'image, un filtre est appliqué en tenant compte de la similarité de couleur et de la distance spatiale des pixels voisins.

#### Parallélisation des boucles

Les boucles suivantes sont parallélisées :

- **Calcul des poids spatiaux** :
  ```cpp
  for (int i = 0; i < d; i++) {
      for (int j = 0; j < d; j++) {
          int x = i - radius, y = j - radius;
          spatial_weights[i * d + j] = gaussian(sqrt(x * x + y * y), sigma_space);
      }
  }
  ```
- **Application du filtre sur l'image** :
  ```cpp
    for (int y = radius; y < height - radius; y++) {
        for (int x = radius; x < width - radius; x++) {
        }
    }
  ```
**Solutions de parallélisation**
Deux approches ont été utilisées pour paralléliser le calcul avec CUDA :

1. Utilisation de deux noyaux distincts
Dans cette approche, deux noyaux CUDA distincts sont utilisés : un pour calculer les poids spatiaux et un autre pour appliquer le filtre sur l'image. Voici l'exemple de code :

**Noyau pour calculer les poids spatiaux** :
```cpp
    dim3 blockSize2D(block_size, block_size);
    dim3 gridSize((d + blockSize2D.x - 1) / blockSize2D.x, (d + blockSize2D.y - 1) / blockSize2D.y);

    calculate_spatial_weights<<<gridSize, blockSize2D>>>(d_spatial_weights, d, sigma_space);
    cudaDeviceSynchronize();
```
**Noyau pour appliquer le filtre bilatéral :**
```cpp
    unsigned char *d_src, *d_dst;
    cudaMalloc(&d_src, width * height * channels * sizeof(unsigned char));
    cudaMalloc(&d_dst, width * height * channels * sizeof(unsigned char));
    cudaMemcpy(d_src, src, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    gridSize = dim3((width + blockSize2D.x - 1) / blockSize2D.x, (height + blockSize2D.y - 1) / blockSize2D.y);

    bilateral_filter_kernel<<<gridSize, blockSize2D>>>(d_src, d_dst, width, height, channels, d, sigma_color, d_spatial_weights);
    cudaDeviceSynchronize();
```
Dans cette approche, deux noyaux sont lancés : l’un calcule les poids spatiaux (calculate_spatial_weights), puis l’autre applique le filtre sur l'image (bilateral_filter_kernel).
2. Utilisation d'un seul noyau
Une autre solution consiste à combiner les deux tâches dans un seul noyau, où le calcul des poids spatiaux et l'application du filtre sont effectués simultanément. Cette approche recalcul donc plusieurs fois certains résultats:
**Noyau unique pour calculer les poids et appliquer le filtre** :
```cpp
dim3 blockSize(block_size, block_size); // Taille des blocs, avec une limite de 1024 threads par bloc
dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y); // Taille de la grille, blockSize*gridSize = 512 pour l'image 

bilateral_filter_kernel<<<gridSize, blockSize>>>(d_src, d_dst, width, height, channels, d, sigma_color, sigma_space);
cudaDeviceSynchronize();
```
Dans cette approche, un seul noyau est utilisé pour effectuer à la fois le calcul des poids spatiaux et l'application du filtre bilatéral.

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
