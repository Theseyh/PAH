#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#pragma pack(push, 1)
typedef struct {
    uint16_t file_type;
    uint32_t file_size;
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t offset;
} BMPHeader;

typedef struct {
    uint32_t header_size;
    int32_t width;
    int32_t height;
    uint16_t planes;
    uint16_t bits_per_pixel;
    uint32_t compression;
    uint32_t image_size;
    int32_t x_pixels_per_m;
    int32_t y_pixels_per_m;
    uint32_t colors_used;
    uint32_t colors_important;
} BMPInfoHeader;
#pragma pack(pop)

void read_bmp(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Erreur lors de l'ouverture du fichier\n");
        return;
    }

    // Lire l'en-tête du fichier
    BMPHeader bmp_header;
    fread(&bmp_header, sizeof(BMPHeader), 1, file);

    // Vérifier que le fichier est bien un BMP
    if (bmp_header.file_type != 0x4D42) {
        printf("Ce fichier n'est pas un fichier BMP valide\n");
        fclose(file);
        return;
    }

    // Lire l'en-tête d'information
    BMPInfoHeader bmp_info_header;
    fread(&bmp_info_header, sizeof(BMPInfoHeader), 1, file);

    // Vérifier que l'image est en niveaux de gris (8 bits par pixel)
    if (bmp_info_header.bits_per_pixel != 8) {
        printf("L'image n'est pas en noir et blanc (doit être en 8 bits par pixel)\n");
        fclose(file);
        return;
    }

    // Aller à l'emplacement des données d'image
    fseek(file, bmp_header.offset, SEEK_SET);

    // Allouer de la mémoire pour les données d'image
    int image_size = bmp_info_header.image_size;
    unsigned char* image_data = (unsigned char*)malloc(image_size);
    if (!image_data) {
        printf("Erreur d'allocation mémoire\n");
        fclose(file);
        return;
    }

    // Lire les données d'image
    fread(image_data, image_size, 1, file);

    // Exemple : Afficher l'intensité du premier pixel
    uint8_t pixel1 = image_data[0];  // Premier pixel (8 bits pour l'intensité)
    printf("Intensité du premier pixel : %d\n", pixel1);

    // Exemple : Afficher l'intensité du second pixel
    uint8_t pixel2 = image_data[1];  // Second pixel
    printf("Intensité du second pixel : %d\n", pixel2);

    // Libérer la mémoire allouée
    free(image_data);
    fclose(file);
}

int main() {
    read_bmp("lena512.bmp");
    return 0;
}

