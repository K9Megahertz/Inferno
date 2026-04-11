#pragma once
#include <numeric>
#include <inferno/util/random.h>
#include <inferno/modules/module.h>


namespace Inferno {

	Tensor Softmax(Tensor& A, int axis = -1);



}