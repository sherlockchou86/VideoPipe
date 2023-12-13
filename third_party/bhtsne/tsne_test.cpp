#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iostream>

#include "tsne.h"

// Function that runs the Barnes-Hut implementation of t-SNE
// simulate input data from code, print result to console
// 3 dims --->  2 dims  for this test
int main() {

    // Define some variables
	int origN = 8, N = 8, D = 3, no_dims = 2, max_iter = 1000;
	double perplexity = 2, theta = 0.5;
    int rand_seed = -1;
	double data[] = {100, 23.9, 12,
					45.2, 46.9, 10,
					201, 44.8, 30,
					201, 40.2, 32,
					101, 7.8, 33,
					21, 4, 35,
					441, 4.8, 36,
					2.1, 32.4, 56};

	N = origN;

	// Now fire up the SNE implementation
	double* Y = (double*) malloc(N * no_dims * sizeof(double));
	if(Y == NULL) { printf("Memory allocation failed!\n"); exit(1); }

	// data: input high dims of features, N * D
	// Y: output low dims of features, N * no_dims
	TSNE::run(data, N, D, Y, no_dims, perplexity, theta, rand_seed, false, max_iter, 250, 250);

	// Print the results
	for (int i = 0; i < N; i++) {
		/* code */
		std::cout << Y[i] << "," << Y[i+1] << std::endl;
	}
	
	// you can display result on 2D screen here

	// Clean up the memory
	free(Y); Y = NULL;
}
