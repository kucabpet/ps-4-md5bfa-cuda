#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cmath>

#define COUNT_HASH 32
#define COUNT_UNIT8_T_HASH 16
#define ALPHABET_COUNT 26

#define DEBUG
#define DEBUG_SHOW_GENERATING_WORD false
#define DEBUG_SHOW_WORDS_ARRAY true
#define DEBUG_SHOW_WORDS_HASHES_ARRAY true

// Declaration
void run_mult(char *words, int height, int width, uint8_t *hashed_words);


void generate_alphabet(char *input) {
    int i = 0;

    for (char c = 'a'; c <= 'z'; c++) {
        input[i++] = c;
    }

    input[i] = '\0';
}

void show_hash(uint8_t *input) {

    for (int i = 0; i < COUNT_UNIT8_T_HASH; i++)
        printf("%2.2x", input[i]);

    printf("\n");
}

int equals_array(uint8_t *array_1, uint8_t *array_2) {
    for (int i = 0; i < COUNT_UNIT8_T_HASH; i++) {
        if (array_1[i] != array_2[i]) {
            return 0;
        }
    }
    return 1;
}

void parse_input_data(char *input, uint8_t *output) {

    char current[2];
    int j = 0;

    for (int i = 0; i < COUNT_HASH; i += 2) {
        current[0] = input[i];
        current[1] = input[i + 1];

        uint8_t value = (uint8_t) strtol(current, NULL, COUNT_UNIT8_T_HASH);
        output[j++] = value;
    }

}

void generate_words(char *prefix, int level, const int max_depth, const char *alphabet,
                    char *words, int *curr_index, const int height, const int width) {
    char curr_word[width];

    level += 1;

    for (int i = 0; i < ALPHABET_COUNT; i++) {
        strcpy(curr_word, "");
        strcpy(curr_word, prefix);
        strncat(curr_word, &alphabet[i], 1);

        if (DEBUG_SHOW_GENERATING_WORD) {
            printf("Generating... %s... \n", curr_word);
        }

        for (int j = 0; j < width; j++) {
            words[width * (*curr_index) + j] = curr_word[j];
        }


        if (*curr_index < height * width) {
            (*curr_index)++;
        }


        if (level < max_depth) {
            generate_words(curr_word, level, max_depth, alphabet, words, curr_index, height, width);
        }
    }
}

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

    int len = atoi(argv[2]);

    printf("Input hash: %s \n", input_data);
    printf("Count of character: %d \n", argv[2]);

    uint8_t input_data_hexa[16];
    parse_input_data(input_data, input_data_hexa);

    char alphabet[ALPHABET_COUNT];
    generate_alphabet(alphabet);

    int height;
    if (len > 1) {
        height = (int) pow(ALPHABET_COUNT, len) + ALPHABET_COUNT;
    } else {
        height = ALPHABET_COUNT;
    }
    int width = len;

#ifdef DEBUG
    printf("debug: height=%d \n", height);
    printf("debug: width=%d \n", width);
#endif

    char *words = new char[width * height];

    int words_index_value = 0;
    int *words_index = &words_index_value;

    generate_words((char *) "", 0, len, alphabet, words, words_index, height, width);

    if (DEBUG_SHOW_WORDS_ARRAY) {
        int x = 0;

        for (int i = 0; i < height * width; i += width) {
            char curr_word[width];

            int j;
            for (j = 0; j < width; j++) {
                curr_word[j] = words[i + j];
            }
            curr_word[j] = '\0';

            printf("[%d]: %s \n", x++, curr_word);
        }
    }

    uint8_t *hashed_words = new uint8_t[COUNT_UNIT8_T_HASH * height];

    // CUDA
    run_mult(words, height, width, hashed_words);

    if (DEBUG_SHOW_WORDS_HASHES_ARRAY) {
        for (int i = 0, j = 0; i < height * width && j < height; i += width, j++) {
            char curr_word[width];

            int x;
            for (x = 0; x < width; x++) {
                curr_word[x] = words[i + x];
            }
            curr_word[x] = '\0';

            printf("%s - ", curr_word);
            show_hash(&hashed_words[COUNT_UNIT8_T_HASH * j]);
        }
    }


    for (int i = 0, j = 0; i < height * width && j < height; i += width, j++) {

        if (equals_array(input_data_hexa, &hashed_words[COUNT_UNIT8_T_HASH * j])) {

            printf("Found match! \n");
            char w[width];

            int x;
            for (x = 0; x < width; x++) {
                w[x] = words[i + x];
            }
            w[x] = '\0';

            printf("Input string: %s \n", w);
        }
    }


    words = NULL;
    delete words;

    printf("\n\nProgram exit\n");
    return 0;
}
