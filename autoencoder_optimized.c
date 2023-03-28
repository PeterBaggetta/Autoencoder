#include <stdlib.h>
#include <stdio.h>
#include <math.h>
// #include <conio.h>
#include <string.h>
#include <time.h>

#define numInputs 784
#define numHiddenNodes 32
#define numOutputs 784
#define numTrainingSets 1000
#define numClasses 10
double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}
double dSigmoid(double x) {
    return x * (1 - x);
}

double relu(double x) {
  if (x <= 0)
      return 0.3*x;
  else
      return x;
}
double dRelu(double x) {
    if (x <= 0)
        return 0.3;
    else
        return 1;
}
double init_weights() {

    double a = (double)rand();
    printf("%g ", a);
    double b = (double)RAND_MAX;
    printf("%g ", b);
    printf("%g ", ((a / b) - 0.5) / 10);
    return ((a / b) - 0.5) / 10;
}
void shuffle(int *array, size_t n){

    if (n > 1) {
        size_t i;

        for (i = 0; i < n-1; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

double get_category(double *y) {
    double i;

    for (i = 0; i < 10; i++) {
        if (y[(int)(i)] == 1) {
          return i;
        }
    }
}

void print_images(double *a0[numTrainingSets],
  double *y[numTrainingSets], int start, int end) {
  for (int train_id = start; train_id < end; train_id++) {
    printf("----- %f -----\n", get_category(y[train_id]));
    for (int row = 0; row < 28; row++) {
      for (int col = 0; col < 28; col++) {
        if (a0[train_id][row * 28 + col] == 0)
            printf("\033[0;37m0.0 ");
        else
            printf("\033[0;31m%.1f ", a0[train_id][row * 28 + col]);
      }
      printf("\033[0;37m\n");
    }
    printf("\n\n\n\n");
  }
}

void print_image(double a[numInputs]) {

    for (int row = 0; row < 28; row++) {
      for (int col = 0; col < 28; col++) {
        if (a[row * 28 + col] < 0.3)
            printf("\033[0;37m%.1f ", a[row * 28 + col]);
        else
            printf("\033[0;31m%.1f ", a[row * 28 + col]);
      }
      printf("\033[0;37m\n");
    }
    printf("\n\n");
}

void print_weights(double *w1[numHiddenNodes], double *b1, double *w2[numOutputs], double *b2) {

    // Print Final Weights after done training
    fputs ("Final Hidden Weights\n[ ", stdout);
    for (int j = 0; j < numHiddenNodes; j++) {
        fputs("[ ", stdout);
        for (int k = 0; k < numInputs; k++) {
            printf("%f ", w1[k][j]);
        }
        fputs("] ", stdout);
    }

    fputs( "]\nFinal Hidden Biases\n[ ", stdout);
    for (int j = 0; j < numHiddenNodes; j++) {
        fputs("[ ", stdout);
        printf("%f ", b1[j]);
        fputs("] ", stdout);
    }

    fputs ("]\nFinal Output Weights\n[ ", stdout);
    for (int j = 0; j < numOutputs; j++) {

        for (int k = 0; k < numHiddenNodes; k++) {
            fputs("[ ", stdout);
            printf("%f ", w2[k][j]);
            fputs("] ", stdout);
        }
    }

    fputs( "]\nFinal Output Biases\n[ ", stdout);
    for (int j = 0; j < numOutputs; j++) {
        fputs("[ ", stdout);
        printf("%f ", b2[j]);
        fputs("] ", stdout);
    }

    fputs ("] \n", stdout);
}




int main(void) {

    clock_t start_t;
    clock_t current_t;
    double time_taken;
    const double lr = 0.01f; // default 0.001

    double *z1 = (double*)malloc(numHiddenNodes * sizeof(double));
    double *z1_test = (double*)malloc(numHiddenNodes * sizeof(double));
    double *z2 = (double*)malloc(numOutputs * sizeof(double));
    double *z2_test = (double*)malloc(numOutputs * sizeof(double));

    double *a1 = (double*)malloc(numHiddenNodes * sizeof(double));
    double *a1_test = (double*)malloc(numHiddenNodes * sizeof(double));
    double *a2 = (double*)malloc(numOutputs * sizeof(double));
    double *a2_test = (double*)malloc(numOutputs * sizeof(double));
    double *e = (double*)malloc(numOutputs * sizeof(double));
    double *e_test = (double*)malloc(numOutputs * sizeof(double));

    double *delta1 = (double*)malloc(numHiddenNodes * sizeof(double));
    double *delta2 = (double*)malloc(numOutputs * sizeof(double));

    double *b1 = (double*)malloc(numHiddenNodes * sizeof(double));
    double *b2 = (double*)malloc(numOutputs * sizeof(double));

    double *w1[numHiddenNodes];
    for (int i = 0; i < numHiddenNodes; i++)
        w1[i] = (double*)malloc(numInputs * sizeof(double));

    double *w2[numOutputs];
    for (int i = 0; i < numOutputs; i++)
        w2[i] = (double*)malloc(numHiddenNodes * sizeof(double));

    double *a0[numTrainingSets];
    for (int i = 0; i < numTrainingSets; i++)
        a0[i] = (double*)malloc(numInputs * sizeof(double));

    double *y[numTrainingSets];
    for (int i = 0; i < numTrainingSets; i++)
        y[i] = (double*)malloc(numClasses * sizeof(double));

    double *a0_test[numTrainingSets];
    for (int i = 0; i < numTrainingSets; i++)
        a0_test[i] = (double*)malloc(numInputs * sizeof(double));

    double *y_test[numTrainingSets];
    for (int i = 0; i < numTrainingSets; i++)
        y_test[i] = (double*)malloc(numClasses * sizeof(double));

    printf("Loading training data...");
    FILE *fp_x = fopen("x_train.csv", "r");
    for (int i = 0; i < numInputs * numTrainingSets; i++)
    {
        fscanf(fp_x, "%lg\n", &a0[(int)(i/numInputs)][i%numInputs]);
    }
    fclose(fp_x);
    FILE *fp_y = fopen("y_train.csv", "r");
    for (int i = 0; i < numClasses * numTrainingSets; i++)
    {
        fscanf(fp_y, "%lg\n", &y[(int)(i/numClasses)][i%numClasses]);
    }
    fclose(fp_y);
    printf("Load Complete.\n");

    printf("Loading testing data...");
    FILE *fp_x_test = fopen("x_test.csv", "r");
    for (int i = 0; i < numInputs * numTrainingSets; i++)
    {
        fscanf(fp_x_test, "%lg\n", &a0_test[(int)(i/numInputs)][i%numInputs]);
    }
    fclose(fp_x_test);
    FILE *fp_y_test = fopen("y_test.csv", "r");
    for (int i = 0; i < numClasses * numTrainingSets; i++)
    {
        fscanf(fp_y_test, "%lg\n", &y_test[(int)(i/numClasses)][i%numClasses]);
    }
    fclose(fp_y_test);
    printf("Load Complete.\n");



    printf("Image samples: \n");
    print_images(a0, y, 0, 10);

    printf("Initializing NN parameters...\n");
    for(int j = 0; j < numInputs; j++) {
        for (int k = 0; k < numHiddenNodes; k++) {
            w1[j][k] = init_weights();
        }
    }

    for (int j = 0; j < numHiddenNodes;j++) {
        for (int k = 0; k < numOutputs; k++) {
            w2[j][k] = init_weights();
        }
    }

    for (int j = 0; j < numHiddenNodes; j++) {
        b1[j] = init_weights();
    }

    for (int j = 0; j < numOutputs; j++) {
        b2[j] = init_weights();
    }
    printf("NN parameter initialization complete.\n");

    int numEpochs = 20;
    double mse[numEpochs];
    double mse_test[numEpochs];

    double mse_training_example;
    double mse_test_example;
    // Train the neural network for a number of epochs
    FILE *fpt;


    printf("Beggining training\n");


    start_t = clock();
    fpt = fopen("results_optimized.csv", "w+");
    fclose(fpt);
    for (int epoch = 0; epoch < numEpochs; epoch ++) {
        printf("epoch %d\n", epoch);
        mse[epoch] = 0;
        // double num_correct_predictions = 0
        // double num_incorrect_predictions = 0;
        for (int i = 0; i < numTrainingSets - 1; i++) {
            // Forward pass

            // Compute hidden layer activation
            for (int j = 0; j < numHiddenNodes; j++) {

                z1[j] = b1[j];

                for (int k = 0; k < numInputs; k++) {
                    z1[j] += a0[i][k] * w1[j][k];
                }
                a1[j] = relu(z1[j]);
            }

            // Compute output layer activation
            for (int j = 0; j < numOutputs; j++) {
                z2[j] = b2[j];

                for (int k = 0; k < numHiddenNodes; k++) {
                    z2[j] += a1[k] * w2[j][k];
                }
                a2[j] = sigmoid(z2[j]);
            }

            // Compute hidden layer activation
            for (int j = 0; j < numHiddenNodes; j++) {

                z1_test[j] = b1[j];

                for (int k = 0; k < numInputs; k++) {
                    z1_test[j] += a0_test[i][k] * w1[j][k];
                }
                a1_test[j] = relu(z1_test[j]);
            }

            // Compute output layer activation
            for (int j = 0; j < numOutputs; j++) {
                z2_test[j] = b2[j];

                for (int k = 0; k < numHiddenNodes; k++) {
                    z2_test[j] += a1_test[k] * w2[j][k];
                }
                a2_test[j] = sigmoid(z2_test[j]);
            }

            if ((i+epoch)%1000 == 0) {
                printf("----- True image -----\n");
                print_image(a0_test[i]);
                printf("----- Autoencoded image -----\n");
                print_image(a2_test);
                printf("\n\n\n\n");
            }

            // Compute errors and mse
            mse_training_example = 0;
            for (int j = 0; j < numOutputs; j++) {
                e[j] = (a0[i][j] - a2[j]);
                mse_training_example += (e[j] * e[j]) / numOutputs;
            }
            mse[epoch] += mse_training_example / numTrainingSets;

            // Compute errors and mse
            mse_test_example = 0;
            for (int j = 0; j < numOutputs; j++) {
                e_test[j] = (a0_test[i][j] - a2_test[j]);
                mse_test_example += (e_test[j] * e_test[j]) / numOutputs;
            }
            mse_test[epoch] += mse_test_example / numTrainingSets;

            // Compute delta2[j]
            for (int j = 0; j < numOutputs; j++) {
                delta2[j] = lr * e[j] * dSigmoid(a2[j]);
            }

            // Compute delta1[j]
            for (int j = 0; j < numHiddenNodes; j++) {
                delta1[j] = 0;
                for (int k = 0; k < numOutputs; k++) {
                    delta1[j] += lr * dRelu(z1[j]) * w2[k][j] * dSigmoid(a2[k]) * e[k];
                }
            }

            // Apply change in output weights
            for (int j = 0; j < numOutputs; j++) {
                b2[j] += delta2[j];
                for (int k = 0; k < numHiddenNodes; k++) {
                    w2[j][k] += a1[k] * delta2[j];
                }
            }

            // Apply change in hidden weights
            for (int j = 0; j < numHiddenNodes; j++) {
                b1[j] += delta1[j];
                for (int k = 0; k < numInputs; k++) {
                    w1[j][k] += a0[i][k] * delta1[j];
                }
            }
        }
        current_t = clock() - start_t;
        time_taken = ((double)current_t) / CLOCKS_PER_SEC;

        printf("%g, %g, %g\n", time_taken, mse[epoch], mse_test[epoch]);
        fpt = fopen("results_optimized.csv", "a+");
        fprintf(fpt,"%g, %g, %g\n", time_taken, mse[epoch], mse_test[epoch]);
        fclose(fpt);
    }

    return 0;
}
