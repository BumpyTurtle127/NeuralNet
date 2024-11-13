#include <stdio.h>
#include <stdlib.h>
#include "fnn.h"

net_fnn * init_net(int numLayers, int * numNodes){
	net_fnn * ret = (net_fnn*)malloc(sizeof(net_fnn));

	ret->weights = (mat_fnn**)malloc((numLayers-1)*sizeof(mat_fnn*));
	for(int i = 0; i < numLayers-1; i++){
		ret->weights[i]
	}
}
