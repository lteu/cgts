import cython
from cython import Py_ssize_t

from libc.math cimport fabs, sqrt, floor
from libc.stdlib cimport free, malloc
from libc.string cimport memmove

import math
import numpy as np 

import numpy as np

cimport numpy as cnp
from numpy cimport (
NPY_FLOAT32,
NPY_FLOAT64,
NPY_INT8,
NPY_INT16,
NPY_INT32,
NPY_INT64,
NPY_OBJECT,
NPY_UINT8,
NPY_UINT16,
NPY_UINT32,
NPY_UINT64,
float32_t,
float64_t,
int8_t,
int16_t,
int32_t,
int64_t,
ndarray,
uint8_t,
uint16_t,
uint32_t,
uint64_t,
)

cnp.import_array()



cdef:
	float64_t FP_ERR = 1e-13
	float64_t NaN = <float64_t>np.NaN
	# int64_t NPY_NAT = get_nat()


# # cimport pandas._libs.util as util
# # from pandas._libs.khash cimport (
# # 	kh_destroy_int64,
# # 	kh_get_int64,
# # 	kh_init_int64,
# # 	kh_int64_t,
# # 	kh_put_int64,
# # 	kh_resize_int64,
# # 	khiter_t,
# # )
# # from pandas._libs.util cimport get_nat, numeric

# # import pandas._libs.missing as missing




# # @cython.wraparound(False)
# # @cython.boundscheck(False)
# # def corrValue(arr1,arr2,n,sum_prod,sum_prod_sqr,x_s_sum,x_sum_s,y_s_sum,y_sum_s):
# # 	for i in range(len(arr1)):

# # 		sum_prod += arr1[i]*arr2[i]
# # 		sum_prod_sqr += (arr1[i]*arr2[i])**2
# # 		x_s_sum += arr1[i]**2
# # 		x_sum_s += arr1[i]
# # 		y_s_sum += arr2[i]**2
# # 		y_sum_s += arr2[i]
# # 		n += 1
# # 		# print('...',arr1[i],arr2[i])

# # 	intermediate = (n*x_s_sum - x_sum_s**2) * (n*y_s_sum - y_sum_s**2)
# # 	if n != 0 and intermediate > 0:
# # 		# pass
# # 		score = (n* sum_prod - x_sum_s*y_sum_s) / math.sqrt( intermediate  )
# # 	else:
# # 		score = 0
# # 	return round(score,4),n,sum_prod,sum_prod_sqr,x_s_sum,x_sum_s,y_s_sum,y_sum_s


@cython.wraparound(False)
@cython.boundscheck(False)
def getFollowingRow(
	const int64_t T,
	const int64_t K, 
	const int64_t step, 
	const int64_t granularity, 
	const int64_t[:, :, :]  in_list_n,
	const float64_t[:, :, :]  in_list_sum_prod, 
	const float64_t[:, :, :]  in_list_x_s_sum, 
	const float64_t[:, :, :]  in_list_x_sum_s, 
	const float64_t[:, :, :]  in_list_y_s_sum, 
	const float64_t[:, :, :]  in_list_y_sum_s,
	const int64_t[:, :, :]  base_list_n,
	const float64_t[:, :, :]  base_list_sum_prod, 
	const float64_t[:, :, :] base_list_x_s_sum, 
	const float64_t[:, :, :] base_list_x_sum_s, 
	const float64_t[:, :, :] base_list_y_s_sum, 
	const float64_t[:, :, :] base_list_y_sum_s
	):
	cdef:
		Py_ssize_t i, j, xi, yi
		# ndarray[float64_t, ndim=2] result
		uint16_t n, base, idx
		int64_t n_step_new
		ndarray[float64_t, ndim=3] result
		float64_t sum_prod,x_s_sum,x_sum_s,y_s_sum,y_sum_s, divisor
		# float64_t divisor1,divisor2
		ndarray[int64_t, ndim=3] list_n
		ndarray[float64_t, ndim=3] list_sum_prod,list_x_s_sum,list_x_sum_s,list_y_s_sum,list_y_sum_s

	
	# base = <int>floor(T/step)
	n_step_new = <int>floor((T-granularity-step)/step)
	result = np.empty((n_step_new, K, K), dtype=np.float64)

	list_n = np.empty((n_step_new, K, K), dtype=np.int64)
	list_sum_prod = np.empty((n_step_new, K, K), dtype=np.float64)
	list_x_s_sum = np.empty((n_step_new, K, K), dtype=np.float64)
	list_x_sum_s = np.empty((n_step_new, K, K), dtype=np.float64)
	list_y_s_sum = np.empty((n_step_new, K, K), dtype=np.float64)
	list_y_sum_s = np.empty((n_step_new, K, K), dtype=np.float64)


	with nogil:
		for i in range(n_step_new):
			for xi in range(K):
				for yi in range(xi + 1):
					idx = (i*step+granularity)/step

					n = in_list_n[i, xi, yi] + base_list_n[idx, xi, yi] 
					sum_prod = in_list_sum_prod[i, xi, yi] + base_list_sum_prod[idx, xi, yi] 
					x_s_sum = in_list_x_s_sum[i, xi, yi] + base_list_x_s_sum[idx, xi, yi] 
					x_sum_s = in_list_x_sum_s[i, xi, yi] + base_list_x_sum_s[idx, xi, yi] 
					y_s_sum = in_list_y_s_sum[i, xi, yi] + base_list_y_s_sum[idx, xi, yi] 
					y_sum_s = in_list_y_sum_s[i, xi, yi] + base_list_y_sum_s[idx, xi, yi] 


					list_n[i, xi, yi] = list_n[i, yi, xi] = n
					list_sum_prod[i, xi, yi] = list_sum_prod[i, yi, xi] = sum_prod
					list_x_s_sum[i, xi, yi] = list_x_s_sum[i, yi, xi] = x_s_sum
					list_x_sum_s[i, xi, yi] = list_x_sum_s[i, yi, xi] = x_sum_s
					list_y_s_sum[i, xi, yi] = list_y_s_sum[i, yi, xi] = y_s_sum
					list_y_sum_s[i, xi, yi] = list_y_sum_s[i, yi, xi] = y_sum_s


					divisor = (n*y_s_sum - y_sum_s**2) * (n*x_s_sum - x_sum_s**2) 
					if n != 0 and divisor > 0:
						result[i, xi, yi] = result[i, yi, xi] = (n* sum_prod - x_sum_s*y_sum_s) / (sqrt( divisor ))
					else:
						result[i, xi, yi] = result[i, yi, xi] = NaN
	return n_step_new, result, list_n,list_sum_prod, list_x_s_sum, list_x_sum_s, list_y_s_sum, list_y_sum_s


@cython.wraparound(False)
@cython.boundscheck(False)
def getFirstRow(const float64_t[:, :] mat, int64_t granularity, int64_t step):
	cdef:
		Py_ssize_t i, j, xi, yi, N, K
		# ndarray[float64_t, ndim=2] result
		uint16_t n,start,idx
		float64_t sum_prod,x_s_sum,x_sum_s,y_s_sum,y_sum_s
		# float64_t divisor1,divisor2
		ndarray[int64_t, ndim=3] list_n
		ndarray[float64_t, ndim=3] list_sum_prod,list_x_s_sum,list_x_sum_s,list_y_s_sum,list_y_sum_s

	N, K = (<object>mat).shape

	list_n = np.empty((step, K, K), dtype=np.int64)
	list_sum_prod = np.empty((step, K, K), dtype=np.float64)
	list_x_s_sum = np.empty((step, K, K), dtype=np.float64)
	list_x_sum_s = np.empty((step, K, K), dtype=np.float64)
	list_y_s_sum = np.empty((step, K, K), dtype=np.float64)
	list_y_sum_s = np.empty((step, K, K), dtype=np.float64)

	# list_n= np.empty((step, K, K), dtype=np.float64)
	# step = <int>floor(K/granularity)

	with nogil:
		for i in range(step):
			start = granularity*i
			for xi in range(K):
				for yi in range(xi + 1):
					n = 0
					sum_prod = 0
					x_s_sum = 0
					x_sum_s = 0
					y_s_sum = 0
					y_sum_s = 0
					for j in range(granularity):
						n = n + 1
						idx = start+j
						sum_prod += mat[idx,xi]*mat[idx,yi]
						x_s_sum += mat[idx,xi]**2
						x_sum_s += mat[idx,xi]
						y_s_sum += mat[idx,yi]**2
						y_sum_s += mat[idx,yi]
					
					list_n[i, xi, yi] = list_n[i, yi, xi] = n
					list_sum_prod[i, xi, yi] = list_sum_prod[i, yi, xi] = sum_prod
					list_x_s_sum[i, xi, yi] = list_x_s_sum[i, yi, xi] = x_s_sum
					list_x_sum_s[i, xi, yi] = list_x_sum_s[i, yi, xi] = x_sum_s
					list_y_s_sum[i, xi, yi] = list_y_s_sum[i, yi, xi] = y_s_sum
					list_y_sum_s[i, xi, yi] = list_y_sum_s[i, yi, xi] = y_sum_s

	return list_n,list_sum_prod, list_x_s_sum, list_x_sum_s, list_y_s_sum, list_y_sum_s
	# result = np.empty((K, K), dtype=np.float64)

	# with nogil:
	# 	for xi in range(K):
	# 		for yi in range(xi + 1):
	# 			n =  list_n[xi,yi]
	# 			sum_prod,x_s_sum = list_sum_prod[xi,yi],list_x_s_sum[xi,yi]				
	# 			x_sum_s,y_s_sum,y_sum_s = list_x_sum_s[xi,yi],list_y_s_sum[xi,yi],list_y_sum_s[xi,yi]
	# 			for i in range(N):
	# 				sum_prod += mat[i,xi]*mat[i,yi]
	# 				x_s_sum += mat[i,xi]**2
	# 				x_sum_s += mat[i,xi]
	# 				y_s_sum += mat[i,yi]**2
	# 				y_sum_s += mat[i,yi]
	# 				n += 1
	# 			divisor1 = (n*x_s_sum - x_sum_s**2) 
	# 			divisor2 = (n*y_s_sum - y_sum_s**2) 
	# 			if n != 0 and divisor2 > 0 and  divisor1 > 0:
	# 				result[xi, yi] = result[yi, xi] = (n* sum_prod - x_sum_s*y_sum_s) / (sqrt( divisor1 )*sqrt( divisor2  ))
	# 			else:
	# 				result[xi, yi] = result[yi, xi] = NaN

	# 			list_x_sum_s[xi,yi],list_y_s_sum[xi,yi],list_y_sum_s[xi,yi] = x_sum_s,y_s_sum,y_sum_s
	# 			list_sum_prod[xi,yi],list_x_s_sum[xi,yi] = sum_prod,x_s_sum
	# 			list_n[xi,yi] = n

	# return result, list_n, list_sum_prod, list_x_s_sum, list_x_sum_s, list_y_s_sum, list_y_sum_s




@cython.boundscheck(False)
@cython.wraparound(False)
def nancorr(const float64_t[:, :] mat, bint cov=False, minp=None):
	cdef:
		Py_ssize_t i, j, xi, yi, N, K
		bint minpv
		ndarray[float64_t, ndim=2] result
		ndarray[uint8_t, ndim=2] mask
		int64_t nobs = 0
		float64_t vx, vy, meanx, meany, divisor, prev_meany, prev_meanx, ssqdmx
		float64_t ssqdmy, covxy

	N, K = (<object>mat).shape

	if minp is None:
		minpv = 1
	else:
		minpv = <int>minp

	result = np.empty((K, K), dtype=np.float64)
	mask = np.isfinite(mat).view(np.uint8)

	with nogil:
		for xi in range(K):
			for yi in range(xi + 1):
				# Welford's method for the variance-calculation
				# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
				nobs = ssqdmx = ssqdmy = covxy = meanx = meany = 0
				for i in range(N):
					if mask[i, xi] and mask[i, yi]:
						vx = mat[i, xi]
						vy = mat[i, yi]
						nobs += 1
						prev_meanx = meanx
						prev_meany = meany
						meanx = meanx + 1 / nobs * (vx - meanx)
						meany = meany + 1 / nobs * (vy - meany)
						ssqdmx = ssqdmx + (vx - meanx) * (vx - prev_meanx)
						ssqdmy = ssqdmy + (vy - meany) * (vy - prev_meany)
						covxy = covxy + (vx - meanx) * (vy - prev_meany)

				if nobs < minpv:
					result[xi, yi] = result[yi, xi] = NaN
				else:
					divisor = (nobs - 1.0) if cov else sqrt(ssqdmx * ssqdmy)

					if divisor != 0:
						result[xi, yi] = result[yi, xi] = covxy / divisor
					else:
						result[xi, yi] = result[yi, xi] = NaN

	return result

@cython.wraparound(False)
@cython.boundscheck(False)
def getBaseMatrixAndDerivatesContd(const float64_t[:, :] mat, int64_t[:, :] list_n,
	float64_t[:, :] list_sum_prod, 
	float64_t[:, :] list_x_s_sum, float64_t[:, :] list_x_sum_s,
	float64_t[:, :] list_y_s_sum,float64_t[:, :] list_y_sum_s):
	cdef:
		Py_ssize_t i, j, xi, yi, N, K
		ndarray[float64_t, ndim=2] result
		uint16_t n
		float64_t sum_prod,x_s_sum,x_sum_s,y_s_sum,y_sum_s
		float64_t divisor1,divisor2

	N, K = (<object>mat).shape
	result = np.empty((K, K), dtype=np.float64)

	with nogil:
		for xi in range(K):
			for yi in range(xi + 1):
				n =  list_n[xi,yi]
				sum_prod,x_s_sum = list_sum_prod[xi,yi],list_x_s_sum[xi,yi]				
				x_sum_s,y_s_sum,y_sum_s = list_x_sum_s[xi,yi],list_y_s_sum[xi,yi],list_y_sum_s[xi,yi]
				for i in range(N):
					sum_prod += mat[i,xi]*mat[i,yi]
					x_s_sum += mat[i,xi]**2
					x_sum_s += mat[i,xi]
					y_s_sum += mat[i,yi]**2
					y_sum_s += mat[i,yi]
					n += 1
				divisor1 = (n*x_s_sum - x_sum_s**2) 
				divisor2 = (n*y_s_sum - y_sum_s**2) 
				if n != 0 and divisor2 > 0 and  divisor1 > 0:
					result[xi, yi] = result[yi, xi] = (n* sum_prod - x_sum_s*y_sum_s) / (sqrt( divisor1 )*sqrt( divisor2  ))
				else:
					result[xi, yi] = result[yi, xi] = NaN

				list_x_sum_s[xi,yi],list_y_s_sum[xi,yi],list_y_sum_s[xi,yi] = x_sum_s,y_s_sum,y_sum_s
				list_sum_prod[xi,yi],list_x_s_sum[xi,yi] = sum_prod,x_s_sum
				list_n[xi,yi] = n

	return result, list_n, list_sum_prod, list_x_s_sum, list_x_sum_s, list_y_s_sum, list_y_sum_s



