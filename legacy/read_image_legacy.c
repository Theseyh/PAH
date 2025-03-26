#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>

void modify_brightness(unsigned char *pixels, int width, int height, float factor) {
    int channels = 1;  // BMP en niveaux de gris
    for (int i = 0; i < width * height * channels; i++) {
        int new_value = pixels[i] * factor;
        pixels[i] = (new_value > 255) ? 255 : new_value;
    }
}

int main() {
    int width, height, channels;

    // Charger l'image
    unsigned char *image = stbi_load("lena512.bmp", &width, &height, &channels, 1);  // Chargement en niveaux de gris
    if (!image) {
        printf("Erreur : Impossible de charger l'image.\n");
        return 1;
    }

    // Modifier la luminosité
    modify_brightness(image, width, height, 1.5);

    // Sauvegarder l'image modifiée
    if (!stbi_write_bmp("lena512_bright.bmp", width, height, 1, image)) {
        printf("Erreur : Impossible de sauvegarder l'image.\n");
    }

    // Libérer la mémoire
    stbi_image_free(image);
    printf("Image modifiée et enregistrée sous 'lena512_bright.bmp'.\n");
    return 0;
}
