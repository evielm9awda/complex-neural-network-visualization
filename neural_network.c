#include <raylib.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <complex.h>

#define MAX_LAYERS 10
#define MAX_NEURONS_PER_LAYER 100
#define MAX_CONNECTIONS 10000
#define LEARNING_RATE 0.001
#define MOMENTUM 0.9
#define EPSILON 1e-8
#define BATCH_SIZE 32

typedef struct {
    float complex value;
    float complex bias;
    float complex delta;
    Vector2 position;
} Neuron;

typedef struct {
    int fromLayer;
    int fromNeuron;
    int toLayer;
    int toNeuron;
    float complex weight;
    float complex prevDeltaWeight;
} Connection;

typedef struct {
    int numLayers;
    int neuronsPerLayer[MAX_LAYERS];
    Neuron neurons[MAX_LAYERS][MAX_NEURONS_PER_LAYER];
    Connection connections[MAX_CONNECTIONS];
    int numConnections;
} NeuralNetwork;

float complex activation(float complex z) {
    return ctanf(clogf(1 + cexpf(z)));
}

float complex activationDerivative(float complex z) {
    float complex ez = cexpf(z);
    float complex ez1 = ez + 1;
    return ez * (ez1 * ctanf(clogf(ez1)) + z * ez) / (ez1 * ez1);
}

float complex randomComplex(float min, float max) {
    float real = min + (max - min) * ((float)rand() / RAND_MAX);
    float imag = min + (max - min) * ((float)rand() / RAND_MAX);
    return CMPLX(real, imag);
}

void initializeNeuralNetwork(NeuralNetwork* nn, int numLayers, int* neuronsPerLayer, int screenWidth, int screenHeight) {
    nn->numLayers = numLayers;
    nn->numConnections = 0;

    float layerSpacing = screenWidth / (numLayers + 1);

    for (int i = 0; i < numLayers; i++) {
        nn->neuronsPerLayer[i] = neuronsPerLayer[i];
        float neuronSpacing = screenHeight / (neuronsPerLayer[i] + 1);

        for (int j = 0; j < neuronsPerLayer[i]; j++) {
            nn->neurons[i][j].value = 0;
            nn->neurons[i][j].bias = randomComplex(-1, 1);
            nn->neurons[i][j].delta = 0;
            nn->neurons[i][j].position = (Vector2){(i + 1) * layerSpacing, (j + 1) * neuronSpacing};
        }
    }

    for (int i = 0; i < numLayers - 1; i++) {
        for (int j = 0; j < neuronsPerLayer[i]; j++) {
            for (int k = 0; k < neuronsPerLayer[i + 1]; k++) {
                nn->connections[nn->numConnections].fromLayer = i;
                nn->connections[nn->numConnections].fromNeuron = j;
                nn->connections[nn->numConnections].toLayer = i + 1;
                nn->connections[nn->numConnections].toNeuron = k;
                float complex limit = csqrtf(6.0 / (neuronsPerLayer[i] + neuronsPerLayer[i+1]));
                nn->connections[nn->numConnections].weight = randomComplex(creal(limit), cimag(limit));
                nn->connections[nn->numConnections].prevDeltaWeight = 0;
                nn->numConnections++;
            }
        }
    }
}

void forwardPropagate(NeuralNetwork* nn, float complex* inputs) {
    for (int i = 0; i < nn->neuronsPerLayer[0]; i++) {
        nn->neurons[0][i].value = inputs[i];
    }

    for (int layer = 1; layer < nn->numLayers; layer++) {
        for (int neuron = 0; neuron < nn->neuronsPerLayer[layer]; neuron++) {
            float complex sum = 0;
            for (int i = 0; i < nn->numConnections; i++) {
                if (nn->connections[i].toLayer == layer && nn->connections[i].toNeuron == neuron) {
                    sum += nn->neurons[nn->connections[i].fromLayer][nn->connections[i].fromNeuron].value * nn->connections[i].weight;
                }
            }
            sum += nn->neurons[layer][neuron].bias;
            nn->neurons[layer][neuron].value = activation(sum);
        }
    }
}

void backPropagate(NeuralNetwork* nn, float complex* targets) {
    int outputLayer = nn->numLayers - 1;

    for (int i = 0; i < nn->neuronsPerLayer[outputLayer]; i++) {
        float complex error = targets[i] - nn->neurons[outputLayer][i].value;
        nn->neurons[outputLayer][i].delta = error * activationDerivative(nn->neurons[outputLayer][i].value);
    }

    for (int layer = outputLayer - 1; layer >= 0; layer--) {
        for (int neuron = 0; neuron < nn->neuronsPerLayer[layer]; neuron++) {
            float complex sum = 0;
            for (int i = 0; i < nn->numConnections; i++) {
                if (nn->connections[i].fromLayer == layer && nn->connections[i].fromNeuron == neuron) {
                    sum += nn->connections[i].weight * nn->neurons[nn->connections[i].toLayer][nn->connections[i].toNeuron].delta;
                }
            }
            nn->neurons[layer][neuron].delta = sum * activationDerivative(nn->neurons[layer][neuron].value);
        }
    }

    for (int i = 0; i < nn->numConnections; i++) {
        float complex gradientClipping = 1;
        float complex deltaWeight = LEARNING_RATE * nn->neurons[nn->connections[i].toLayer][nn->connections[i].toNeuron].delta *
                                    conj(nn->neurons[nn->connections[i].fromLayer][nn->connections[i].fromNeuron].value);

        float complex magnitude = cabsf(deltaWeight);
        if (creal(magnitude) > creal(gradientClipping)) {
            deltaWeight *= gradientClipping / magnitude;
        }

        nn->connections[i].weight += deltaWeight + MOMENTUM * nn->connections[i].prevDeltaWeight;
        nn->connections[i].prevDeltaWeight = deltaWeight;
    }

    for (int layer = 1; layer < nn->numLayers; layer++) {
        for (int neuron = 0; neuron < nn->neuronsPerLayer[layer]; neuron++) {
            nn->neurons[layer][neuron].bias += LEARNING_RATE * nn->neurons[layer][neuron].delta;
        }
    }
}

float complex calculateError(NeuralNetwork* nn, float complex* targets) {
    int outputLayer = nn->numLayers - 1;
    float complex error = 0;

    for (int i = 0; i < nn->neuronsPerLayer[outputLayer]; i++) {
        float complex diff = targets[i] - nn->neurons[outputLayer][i].value;
        error += 0.5f * diff * conj(diff);
    }

    return error;
}

void trainNeuralNetwork(NeuralNetwork* nn, float complex** inputs, float complex** targets, int numSamples, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        float complex totalError = 0;

        for (int sample = 0; sample < numSamples; sample += BATCH_SIZE) {
            for (int b = 0; b < BATCH_SIZE && sample + b < numSamples; b++) {
                forwardPropagate(nn, inputs[sample + b]);
                backPropagate(nn, targets[sample + b]);
                totalError += calculateError(nn, targets[sample + b]);
            }
        }

        float complex averageError = totalError / numSamples;

        printf("Epoch %d: Error = %f + %fi\n", epoch + 1, creal(averageError), cimag(averageError));

        if (cabsf(averageError) < EPSILON) {
            printf("Convergence achieved. Training complete.\n");
            break;
        }
    }
}

void drawNeuralNetwork(NeuralNetwork* nn) {
    // Draw connections
    for (int i = 0; i < nn->numConnections; i++) {
        Vector2 start = nn->neurons[nn->connections[i].fromLayer][nn->connections[i].fromNeuron].position;
        Vector2 end = nn->neurons[nn->connections[i].toLayer][nn->connections[i].toNeuron].position;

        float weight_magnitude = cabsf(nn->connections[i].weight);
        float hue = (atan2f(cimag(nn->connections[i].weight), creal(nn->connections[i].weight)) + M_PI) / (2 * M_PI);
        Color lineColor = ColorFromHSV(hue * 360, 1.0f, weight_magnitude);

        DrawLineEx(start, end, 1.0f + weight_magnitude * 2.0f, lineColor);
    }

    // Draw neurons
    int globalNeuronCount = 0;
    for (int layer = 0; layer < nn->numLayers; layer++) {
        for (int neuron = 0; neuron < nn->neuronsPerLayer[layer]; neuron++) {
            Vector2 pos = nn->neurons[layer][neuron].position;
            float activation_magnitude = cabsf(nn->neurons[layer][neuron].value);
            float hue = (atan2f(cimag(nn->neurons[layer][neuron].value), creal(nn->neurons[layer][neuron].value)) + M_PI) / (2 * M_PI);
            Color neuronColor = ColorFromHSV(hue * 360, 1.0f, 0.7f);
            DrawCircleV(pos, 15.0f, neuronColor);
            DrawCircleLines(pos.x, pos.y, 15.0f, BLACK);

            char neuronLabel[10];
            snprintf(neuronLabel, sizeof(neuronLabel), "%d", globalNeuronCount++);
            DrawText(neuronLabel, pos.x - 5, pos.y - 5, 10, BLACK);
        }
    }
}

int main() {
    const int screenWidth = 800;
    const int screenHeight = 600;

    InitWindow(screenWidth, screenHeight, "Neural Network Visualization");
    SetTargetFPS(60);

    SetRandomSeed(time(NULL));

    NeuralNetwork nn;
    int layers[] = {2, 4, 3, 1};
    int numLayers = sizeof(layers) / sizeof(layers[0]);
    initializeNeuralNetwork(&nn, numLayers, layers, screenWidth, screenHeight);

    float complex** inputs = (float complex**)malloc(4 * sizeof(float complex*));
    float complex** targets = (float complex**)malloc(4 * sizeof(float complex*));

    for (int i = 0; i < 4; i++) {
        inputs[i] = (float complex*)malloc(2 * sizeof(float complex));
        targets[i] = (float complex*)malloc(1 * sizeof(float complex));
    }

    // Complex-valued XOR problem
    inputs[0][0] = 0; inputs[0][1] = 0; targets[0][0] = 0;
    inputs[1][0] = 0; inputs[1][1] = 1 + I; targets[1][0] = 1 + I;
    inputs[2][0] = 1 + I; inputs[2][1] = 0; targets[2][0] = 1 + I;
    inputs[3][0] = 1 + I; inputs[3][1] = 1 + I; targets[3][0] = 0;

    trainNeuralNetwork(&nn, inputs, targets, 4, 1000);

    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(RAYWHITE);

        drawNeuralNetwork(&nn);

        DrawText("Neural Network Visualization", 10, 10, 20, BLACK);

        EndDrawing();
    }

    CloseWindow();

    for (int i = 0; i < 4; i++) {
        free(inputs[i]);
        free(targets[i]);
    }
    free(inputs);
    free(targets);

    return 0;
}
