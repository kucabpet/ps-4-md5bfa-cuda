/*
 * Simple MD5 implementation
 *
 * Compile with: gcc -o md5 md5.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cmath>


#define ALPHABET_COUNT 26
#define SHOW_GENERATING false
#define SHOW_WORDS_ARRAY false

void generate_alphabet(char *input) {
    int i = 0;

    for (char c = 'a'; c <= 'z'; c++) {
        input[i++] = c;
    }

    input[i] = '\0';
}

void show_hash(uint8_t *input) {
    // display result
    for (int i = 0; i < 16; i++)
        printf("%2.2x", input[i]);

    puts("\n");
}

int equals_array(uint8_t *array_1, uint8_t *array_2) {
    for (int i = 0; i < 16; i++) {
        if (array_1[i] != array_2[i]) {
            return 0;
        }
    }
    return 1;
}

void parse_input_data(char *input, uint8_t *output) {

    char current[2];
    int j = 0;

    for (int i = 0; i < 32; i += 2) {
        current[0] = input[i];
        current[1] = input[i + 1];

        uint8_t value = (uint8_t) strtol(current, NULL, 16);
        output[j++] = value;
    }

}

void generate_words(char *prefix, int level, const int max_depth, const char *alphabet,
                    char *words, int *curr_index, const int height, const int width) {
    char tmp[width];

    level += 1;

    for (int i = 0; i < ALPHABET_COUNT; i++) {
        strcpy(tmp, "");
        strcpy(tmp, prefix);
        strncat(tmp, &alphabet[i], 1);

        if (SHOW_GENERATING) {
            printf("Generating... %s... \n", tmp);
        }

        for (int j = 0; j < width; j++) {
	    words[width * (*curr_index) + j] = tmp[j];
        }

		
	if (*curr_index < height*width) {
        	(*curr_index)++;
	}
	

        if (level < max_depth) {
            generate_words(tmp, level, max_depth, alphabet, words, curr_index, height, width);
        }
    }
}

// Declaration
void run_mult(char *words, int height, int width);


int main(int argc, char **argv) {

    if (argc < 3) {
        printf("usage: %s <hash>, <count of letter>\n", argv[0]);
        return 1;

    }

    char *input_data = argv[1];

    if (strlen(input_data) != 32) {
        printf("input hash must have 32 character length");
        return 1;
    }

    int len;
    len = atoi(argv[2]);

    printf("Input hash: %s \n", input_data);
    printf("Count of character: %d \n", len);

    uint8_t input_data_hexa[16];
    parse_input_data(input_data, input_data_hexa);

    char alphabet[ALPHABET_COUNT];
    generate_alphabet(alphabet);

    int height;
    if (len == 1) {
        height = ALPHABET_COUNT;
    } else {
        height = pow(ALPHABET_COUNT, len) + ALPHABET_COUNT;
    }
    int width = len;

    printf("debug: height=%d \n", height);
    printf("debug: width=%d \n", width);

    char *words = new char[width * height];

    int words_index_value = 0;
    int *words_index = &words_index_value;

    generate_words((char *) "", 0, len, alphabet, words, words_index, height, width);

    if (SHOW_WORDS_ARRAY) {
        int x = 0;
        printf("\n Resutl: \n");

        for (int i = 0; i < height * width; i += width) {
            char w[width];

            int j;
            for (j = 0; j < width; j++) {
                w[j] = words[i + j];
            }
            w[j] = '\0';

            printf("[%d]: %s \n", x++, w);
        }
    }

    run_mult(words, height, width);

    words = NULL;
    delete words;

    printf("\n\n Program exit \n");
    return 0;
}
