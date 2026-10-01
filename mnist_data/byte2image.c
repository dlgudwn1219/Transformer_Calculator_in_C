#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

// Big Endian to Little Endian
uint32_t swap_endian(uint32_t val){
    return ((val << 24) & 0xff000000) |
            ((val << 8) & 0x00ff0000) |
            ((val >> 8) & 0x0000ff00) |
            ((val >> 24) & 0x000000ff);
}

int main(int argc, char *argv[]) {

    // 1. Read file in binary mode
    FILE *file = fopen("train-images-idx3-ubyte", "rb");
    if (file == NULL) {
        printf("Cannot open file.. check if download is done correctly, or if file name is correct");
        return 1;
    }

    uint32_t magic_number, num_images, num_rows, num_cols;

    // 2. Read 4bytes(32bits) 4 times for each variable
    fread(&magic_number, sizeof(uint32_t), 1, file);
    fread(&num_images, sizeof(uint32_t), 1, file);
    fread(&num_rows, sizeof(uint32_t), 1, file);
    fread(&num_cols, sizeof(uint32_t), 1, file);

    // 3. Swap big-endian to little-endian
    magic_number = swap_endian(magic_number);
    num_images = swap_endian(num_images);
    num_rows = swap_endian(num_rows);
    num_cols = swap_endian(num_cols);

    // 4. Print
    printf("Magic number(2051 for image, 2049 for label): %d\n", magic_number);
    printf("# images: %d\n", num_images);
    printf("rows: %d, cols: %d\n", num_cols, num_rows);

    // 5. Get argument
    int image_idx;
    if (argc < 2) image_idx = 0;
    else image_idx = atoi(argv[1]);

    printf("Reading image #%d\n", image_idx);

    // 5. Print Image to terminal
    unsigned char image[28][28];
    fseek(file, 28 * 28 * image_idx, SEEK_CUR);
    fread(image, sizeof(unsigned char), 28 * 28, file);
    
    for (int i = 0; i < 28; i++){
        for (int j = 0; j < 28; j++){
            if (image[i][j] > 200) printf("@");
            else if (image[i][j] > 100) printf("*");
            else if (image[i][j] > 50) printf(".");
            else printf(" ");
        }
        printf("\n");
    }

    // 6. Make a PGM text image file
    
    FILE *pgm_file = fopen("output.pgm", "w");
    fprintf(pgm_file, "P2\n");
    fprintf(pgm_file, "28 28\n");
    fprintf(pgm_file, "255\n");

    for (int i = 0; i < 28; i++){
        for (int j = 0; j < 28; j++){
            fprintf(pgm_file, "%d ", image[i][j]);
        }
        fprintf(pgm_file, "\n");
    }

    fclose(pgm_file);
    printf("output.pgm file created\n");

    return 0;
}
