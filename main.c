#include "mainHeader.h"
#include "train_data.h"

#define LEARNING_RATE 1

double weights[ROW][COL] = {
	{0.0, 0.0, 0.0, 0.0, 0.0},
	{0.0, 0.0, 0.0, 0.0, 0.0},
	{0.0, 0.0, 0.0, 0.0, 0.0},
	{0.0, 0.0, 0.0, 0.0, 0.0},
	{0.0, 0.0, 0.0, 0.0, 0.0}
};

double delta_weights[ROW][COL] = {
	{0.0, 0.0, 0.0, 0.0, 0.0},
	{0.0, 0.0, 0.0, 0.0, 0.0},
	{0.0, 0.0, 0.0, 0.0, 0.0},
	{0.0, 0.0, 0.0, 0.0, 0.0},
	{0.0, 0.0, 0.0, 0.0, 0.0}
};


// functions to print the weights and delta weights
void printWeights(){
	int i, j;

	printf("{\n");
	for(i= 0; i<ROW; i++){
		printf("\t{");
		for(j = 0; j<COL; j++){
			printf("%.1lf", weights[i][j]);

			if(j+1!=ROW){
				printf(", ");
			}
		}

		printf("},\n");
	}

	printf("}");
}

void printDeltaWeights(){
	int i, j;

	printf("{\n");
	for(i= 0; i<ROW; i++){
		printf("\t{");
		for(j = 0; j<COL; j++){
			printf("%.1lf", delta_weights[i][j]);

			if(j+1!=ROW){
				printf(", ");
			}
		}

		printf("},\n");
	}

	printf("}");
}

// function for an activation function (Binary step 0 activation)
int activation(int sum){
	if(sum<=0){
		return 0;
	}

	return 1;
}


// Function to change the value of weights after calculating delta weights
void learn(){
	int i, j;

	for(i = 0; i<ROW; i++){
		for(j = 0; j<COL; j++){
			weights[i][j] = weights[i][j] + delta_weights[i][j];
		}
	}
}


// Train function to start the training process
void train(){
	double sum = 0.0;
	int i, j, k;
	int y_result_activated;
	int indexLabels = 0;

	while(true){
		system("cls");
		indexLabels = 0;
		sum = 0.0;

		printf("Weights:  \n"); printWeights();
		printf("\n\n");
		printf("D Weights: \n"); printDeltaWeights();
		printf("\n");

		for(i = 0; i<JMLDATA; i++){

			for(j = 0; j<ROW; j++){
				for(k = 0; k<COL; k++){
					sum += trainData_feature[i][j][k] * weights[j][k];
				}
			}

			y_result_activated = activation(sum);

			// printf("\n\nY: %d\n\n", y_result_activated);

			for(j = 0; j<ROW; j++){
				for(k = 0; k<COL; k++){
					delta_weights[j][k] = (labels[indexLabels] - y_result_activated) * LEARNING_RATE * trainData_feature[i][j][k];
				}
			}

			learn();

			indexLabels++;
		}

		// usleep(10000);
		sleep(1);
	}
	
}

int main(int argc, char *argv[]) {
	
	train();
	
	return 0;
}
