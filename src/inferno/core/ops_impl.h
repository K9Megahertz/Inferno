#include <inferno/core/tensor.h>

namespace Inferno {


	void SetImpl(Tensor& t, std::shared_ptr<TensorImpl> impl);
	std::shared_ptr<TensorImpl> GetImpl(Tensor& t);
	std::shared_ptr<TensorImpl> GetImpl(const Tensor& t);


	Tensor matmul_impl(const Tensor& A, const Tensor& B, std::string label = "Unlabeled");
	Tensor transpose_impl(const Tensor& A, int dima, int dimb);
	Tensor reshape_impl(const Tensor& A, const std::vector<size_t>& newshape);	
	Tensor contiguous_impl(const Tensor& A);	
	Tensor slice_impl(const Tensor& A, int axis, const size_t start, const size_t end, const size_t step);
	
	Tensor make_view(const Tensor& base, const std::vector<size_t>& new_shape, const std::vector<size_t>& new_strides, size_t new_storage_offset, const std::string& name);


}