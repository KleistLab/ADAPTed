"""
ADAPTed (Adapter and poly(A) Detection And Profiling Tool)

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de
"""

cimport cython

import numpy as np

cimport numpy as np
from libc.math cimport log

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

DTYPE_INT = np.int64
ctypedef np.int64_t DTYPE_INT_t

# Calculate the variance of a segment of signal, given cumsum and cumsum of squared signal.
@cython.boundscheck(False)
cdef inline double var_c(DTYPE_INT_t start, DTYPE_INT_t end, 
						 DTYPE_t[:] c, DTYPE_t[:] c2):
	"""
	Adapted from https://github.com/jmschrei/PyPore/blob/master/PyPore/cparsers.pyx

	The MIT License (MIT)

	Copyright (c) 2014 jmschrei
	"""
	
	if start == end:
		return 0
	if start == 0:
		return c2[end-1]/end - (c[end-1]/end) ** 2
	return (c2[end-1]-c2[start-1])/(end-start) - ((c[end-1]-c[start-1])/(end-start)) ** 2

# find single best split based on maximal LLR
def _best_split(DTYPE_INT_t start, 
				DTYPE_INT_t end, 
				np.ndarray[DTYPE_t] c, 
				np.ndarray[DTYPE_t] c2,
				DTYPE_INT_t offset_head, 
				DTYPE_INT_t offset_tail):
	
	cdef DTYPE_t var_summed
	cdef DTYPE_t var_summed_head
	cdef DTYPE_t var_summed_tail
	cdef DTYPE_t gain
	cdef DTYPE_t split_gain = 0.
	cdef DTYPE_INT_t x = -1
	cdef DTYPE_INT_t i
	
	var_summed = ( end-start ) * log( var_c( start, end, c, c2))
	for i in range( start + offset_head, end - offset_tail):
		var_summed_head = ( i-start ) * log( var_c( start, i, c, c2 ) )
		var_summed_tail = ( end-i ) * log( var_c( i, end, c, c2 ) )
		gain = var_summed-( var_summed_head+var_summed_tail )
		if gain > split_gain:
			split_gain = gain
			x = i
	
	return x, split_gain

# return all LLRs
def _gains(DTYPE_INT_t start, 
				DTYPE_INT_t end, 
				np.ndarray[DTYPE_t] c, 
				np.ndarray[DTYPE_t] c2, 
				DTYPE_INT_t offset_head, 
				DTYPE_INT_t offset_tail,
				DTYPE_INT_t stride = 1):
	
	cdef DTYPE_t var_summed
	cdef DTYPE_t var_summed_head
	cdef DTYPE_t var_summed_tail
	cdef DTYPE_INT_t i

	cdef np.ndarray[DTYPE_t] gains = np.zeros_like(c)
	
	var_summed = ( end-start ) * log( var_c( start, end, c, c2))
	for i in range( start + offset_head, end - offset_tail, stride):
		var_summed_head = ( i-start ) * log( var_c( start, i, c, c2 ) )
		var_summed_tail = ( end-i ) * log( var_c( i, end, c, c2 ) )
		gains[i] = var_summed-( var_summed_head+var_summed_tail )

	return gains


def _gains_w_early_stop(DTYPE_INT_t start, 
				DTYPE_INT_t end, 
				np.ndarray[DTYPE_t] c, 
				np.ndarray[DTYPE_t] c2, 
				DTYPE_INT_t offset_head, 
				DTYPE_INT_t offset_tail,
				DTYPE_INT_t stride = 1,
				DTYPE_INT_t early_stop_window = 500,
				DTYPE_INT_t early_stop_stride = 100):
	
	# for early stopping to work, stride needs to be a divisor of early_stop_stride
	assert early_stop_stride % stride == 0

	cdef DTYPE_t var_summed
	cdef DTYPE_t var_summed_head
	cdef DTYPE_t var_summed_tail
	cdef DTYPE_INT_t i

	cdef np.ndarray[DTYPE_t] gains = np.zeros_like(c)

	var_summed = ( end-start ) * log( var_c( start, end, c, c2))
	for i in range( start + offset_head, end - offset_tail, stride):
		
		if (i >= start + offset_head + early_stop_window) and ((i - (start + offset_head)) % early_stop_stride == 0):
			derivatives = np.diff(gains[i-early_stop_window:i:stride])
			if derivatives.mean() < 0:
				break

		var_summed_head = ( i-start ) * log( var_c( start, i, c, c2 ) )
		var_summed_tail = ( end-i ) * log( var_c( i, end, c, c2 ) )
		gains[i] = var_summed-( var_summed_head+var_summed_tail )

	return gains


def _gains_w_polya_early_stop(DTYPE_INT_t start, 
				DTYPE_INT_t end, 
				np.ndarray[DTYPE_t] c, 
				np.ndarray[DTYPE_t] c2, 
				DTYPE_INT_t offset_head, 
				DTYPE_INT_t offset_tail,
				DTYPE_INT_t stride = 1,
				DTYPE_INT_t adapter_early_stop_window = 1000,
				DTYPE_INT_t adapter_early_stop_stride = 500,
				DTYPE_INT_t polya_early_stop_window = 50,
				DTYPE_INT_t polya_early_stop_stride = 10):
	
	# for early stopping to work, stride needs to be a divisor of early_stop_stride
	assert adapter_early_stop_stride % stride == 0
	assert polya_early_stop_stride % stride == 0

	cdef DTYPE_t var_summed
	cdef DTYPE_t var_summed_head
	cdef DTYPE_t var_summed_tail
	cdef DTYPE_INT_t i

	cdef np.ndarray[DTYPE_t] gains = np.zeros_like(c)

	cdef DTYPE_INT_t adapter_found = 0
	cdef DTYPE_INT_t polya_found = 0

	var_summed = ( end-start ) * log( var_c( start, end, c, c2))
	for i in range( start + offset_head, end - offset_tail, stride):
		
		if ((not adapter_found) 
			and (i >= start + offset_head + adapter_early_stop_window) 
			and ((i - (start + offset_head)) % adapter_early_stop_stride == 0)):

			derivatives = np.diff(gains[i-adapter_early_stop_window:i:stride])
			if derivatives.mean() < 0:
				adapter_found = 1
		
		if (adapter_found) and (not polya_found):
			derivatives = np.diff(gains[i-polya_early_stop_window:i:stride])
			if derivatives.mean() > 0:
				polya_found = 1
				break

		var_summed_head = ( i-start ) * log( var_c( start, i, c, c2 ) )
		var_summed_tail = ( end-i ) * log( var_c( i, end, c, c2 ) )
		gains[i] = var_summed-( var_summed_head+var_summed_tail )

	return gains




def c_llr_trace(np.ndarray[DTYPE_t] raw_signal, 
		  DTYPE_INT_t start, 
		  DTYPE_INT_t end, 
		  DTYPE_INT_t min_obs, 
		  DTYPE_INT_t border_trim,
		  DTYPE_INT_t stride = 1,
		  DTYPE_INT_t adapter_early_stopping = 0,
		  DTYPE_INT_t adapter_early_stop_window = 500,
		  DTYPE_INT_t adapter_early_stop_stride = 100,
		  DTYPE_INT_t polya_early_stopping = 0,
		  DTYPE_INT_t polya_early_stop_window = 50,
		  DTYPE_INT_t polya_early_stop_stride = 10,
		  DTYPE_INT_t return_c_c2 = 0):

	cdef np.ndarray[DTYPE_t] c = np.cumsum( raw_signal )
	cdef np.ndarray[DTYPE_t] c2 = np.cumsum( np.multiply( raw_signal, raw_signal ) )

	cdef np.ndarray[DTYPE_t] gain

	if polya_early_stopping > 0:
		gain = _gains_w_polya_early_stop(start, end, c, c2, min_obs, border_trim, stride, adapter_early_stop_window, adapter_early_stop_stride, polya_early_stop_window, polya_early_stop_stride)
	elif adapter_early_stopping > 0:
		gain = _gains_w_early_stop(start, end, c, c2, min_obs, border_trim, stride, adapter_early_stop_window, adapter_early_stop_stride)
	else:
		gain = _gains(start, end, c, c2, min_obs, border_trim, stride)

	if return_c_c2:
		return gain, c, c2
	else:
		return gain

# TODO: refactor adapter and poly with common base function
def c_llr_detect_adapter(np.ndarray[DTYPE_t] raw_signal, 
						 DTYPE_INT_t min_obs_adapter,
						 DTYPE_INT_t border_trim):

	cdef np.ndarray[DTYPE_t] c = np.cumsum( raw_signal )
	cdef np.ndarray[DTYPE_t] c2 = np.cumsum( np.multiply( raw_signal, raw_signal ) )

	cdef DTYPE_INT_t x_first = 0
	cdef DTYPE_INT_t x_head = 0
	cdef DTYPE_INT_t x_tail = 0
	cdef DTYPE_INT_t length = len(raw_signal ) - 1
	cdef DTYPE_t gain_head = 0.
	cdef DTYPE_t gain_tail = 0.

	x_first, _ = _best_split(0, length, c, c2, min_obs_adapter + border_trim, border_trim)
	x_head, gain_head = _best_split(0, x_first, c, c2, border_trim, min_obs_adapter)
	x_tail, gain_tail = _best_split(x_first, length, c, c2, min_obs_adapter, border_trim)

	if x_first == -1:
		# empty signal
		return 0,0
	if x_head == -1:
		#x_first - border_trim - min_obs_adapter == 0
		x_head = 1
	if x_tail == -1:
		#(length - x_first) - border_trim - min_obs_adapter == 0
		x_tail = x_first+1

		
	cdef np.ndarray[DTYPE_t] medians = np.zeros(4)
	medians[0] = np.median(raw_signal[:x_head])
	medians[1] = np.median(raw_signal[x_head: x_first])
	medians[2] = np.median(raw_signal[x_first:x_tail])
	medians[3] = np.median(raw_signal[x_tail:])

	
	cdef np.ndarray[DTYPE_t] diffs = np.diff(medians)

	# use fact that adapter represents a drop in pA space
	if diffs[1] > 0: # first detected end of adapter
		# full adapter: open_pore/prev_RNA - DNA - RNA - RNA
		if medians[0] >= medians.mean(): 
			return x_head, x_first
		# partial adapter: DNA - RNA - RNA - RNA
		else:
			return 0, x_first 
	elif gain_tail > gain_head: # first detected start of adapter
		return x_first, x_tail
	else:
		return 0,0

def c_llr_detect_adapter_polya(np.ndarray[DTYPE_t] raw_signal, 
							   DTYPE_INT_t min_obs_adapter,
							   DTYPE_INT_t border_trim,
							   DTYPE_INT_t min_obs_polya):

	cdef np.ndarray[DTYPE_t] c = np.cumsum( raw_signal )
	cdef np.ndarray[DTYPE_t] c2 = np.cumsum( np.multiply( raw_signal, raw_signal ) )

	cdef DTYPE_INT_t x_first = 0
	cdef DTYPE_INT_t x_head = 0
	cdef DTYPE_INT_t x_tail = 0
	cdef DTYPE_INT_t length = len(raw_signal ) - 1
	cdef DTYPE_t gain_head = 0.
	cdef DTYPE_t gain_tail = 0.

	x_first, _ = _best_split(0, length, c, c2, min_obs_adapter + border_trim, border_trim)
	x_head, gain_head = _best_split(0, x_first, c, c2, border_trim, min_obs_adapter)
	x_tail, gain_tail = _best_split(x_first, length, c, c2, min_obs_adapter, border_trim)

	if x_first == -1:
		# empty signal
		return 0,0
	if x_head == -1:
		#x_first - border_trim - min_obs_adapter == 0
		x_head = 1
	if x_tail == -1:
		#(length - x_first) - border_trim - min_obs_adapter == 0
		x_tail = x_first+1

		
	cdef np.ndarray[DTYPE_t] medians = np.zeros(4)
	medians[0] = np.median(raw_signal[:x_head])
	medians[1] = np.median(raw_signal[x_head: x_first])
	medians[2] = np.median(raw_signal[x_first:x_tail])
	medians[3] = np.median(raw_signal[x_tail:])

	
	cdef np.ndarray[DTYPE_t] diffs = np.diff(medians)

	cdef DTYPE_INT_t adapter_start = 0
	cdef DTYPE_INT_t adapter_end = 0
	cdef DTYPE_INT_t polya_end = 0

	# use fact that adapter represents a drop in pA space
	if diffs[1] > 0: # first detected end of adapter
		# full adapter: open_pore/prev_RNA - DNA - RNA - RNA
		if medians[0] >= medians.mean(): 
			adapter_start = x_head
			adapter_end = x_first
			# TODO: use x_tail if consistent with min_obs_polya

		# partial adapter: DNA - RNA - RNA - RNA
		else:
			adapter_start = 0
			adapter_end = x_first
	elif gain_tail > gain_head: # first detected start of adapter
		adapter_start = x_first
		adapter_end = x_tail

	else:
		adapter_start = 0
		adapter_end = 0

	if adapter_end == 0:
		return 0,0, 0 # no adapter found

	# find polyA tail
	polya_end, _ = _best_split(adapter_end, length, c, c2, min_obs_polya, border_trim)

	if polya_end == -1:
		#(length - adapter_end) - border_trim - min_obs_polya == 0
		polya_end = 0 # no polyA tail found

	return adapter_start, adapter_end, polya_end

	


def c_llr_detect_adapter_trace(np.ndarray[DTYPE_t] raw_signal, 
						       DTYPE_INT_t min_obs_adapter,
						       DTYPE_INT_t border_trim):

	cdef np.ndarray[DTYPE_t] c = np.cumsum( raw_signal )
	cdef np.ndarray[DTYPE_t] c2 = np.cumsum( np.multiply( raw_signal, raw_signal ) )

	cdef DTYPE_INT_t x_first = 0
	cdef DTYPE_INT_t length = len(raw_signal ) - 1
	cdef np.ndarray[DTYPE_t] gains_first 
	cdef np.ndarray[DTYPE_t] gains_head 
	cdef np.ndarray[DTYPE_t] gains_tail 

	gains_first = _gains(0, length, c, c2, min_obs_adapter + border_trim, border_trim)
	x_first = np.argmax( gains_first )
	gains_head = _gains(0, x_first, c, c2, border_trim, min_obs_adapter)
	gains_tail = _gains(x_first, length, c, c2, min_obs_adapter, border_trim)

	return gains_first, gains_head, gains_tail

def c_llr_detect_adapter_polya_trace(np.ndarray[DTYPE_t] raw_signal, 
						       DTYPE_INT_t min_obs_adapter,
						       DTYPE_INT_t border_trim,
							   DTYPE_INT_t min_obs_polya):

	cdef np.ndarray[DTYPE_t] c = np.cumsum( raw_signal )
	cdef np.ndarray[DTYPE_t] c2 = np.cumsum( np.multiply( raw_signal, raw_signal ) )

	cdef DTYPE_INT_t x_first = 0
	cdef DTYPE_INT_t x_last = 0
	cdef DTYPE_INT_t length = len(raw_signal ) - 1
	cdef np.ndarray[DTYPE_t] gains_first 
	cdef np.ndarray[DTYPE_t] gains_head 
	cdef np.ndarray[DTYPE_t] gains_tail 
	cdef np.ndarray[DTYPE_t] gains_polya

	gains_first = _gains(0, length, c, c2, min_obs_adapter + border_trim, border_trim)
	x_first = np.argmax( gains_first ) # first segmentation point
	gains_head = _gains(0, x_first, c, c2, border_trim, min_obs_adapter)
	gains_tail = _gains(x_first, length, c, c2, min_obs_adapter, border_trim)
	x_last = np.argmax( gains_tail )

	gains_polya = _gains(x_last, length, c, c2, min_obs_polya, border_trim)

	return gains_first, gains_head, gains_tail, gains_polya


def c_llr_boundary_traces(np.ndarray[DTYPE_t] raw_signal, 
				DTYPE_INT_t min_obs_adapter,
				DTYPE_INT_t border_trim):

	cdef np.ndarray[DTYPE_t] c = np.cumsum( raw_signal )
	cdef np.ndarray[DTYPE_t] c2 = np.cumsum( np.multiply( raw_signal, raw_signal ) )

	cdef DTYPE_INT_t x_first = 0
	cdef DTYPE_INT_t length = len(raw_signal ) - 1
	cdef np.ndarray[DTYPE_t] gains_first 
	cdef np.ndarray[DTYPE_t] gains_head 
	cdef np.ndarray[DTYPE_t] gains_tail 

	gains_first = _gains(0, length, c, c2, min_obs_adapter + border_trim, border_trim)
	x_first = np.argmax( gains_first ) # first segmentation point
	gains_head = _gains(0, x_first, c, c2, border_trim, min_obs_adapter)
	gains_tail = _gains(x_first, length, c, c2, min_obs_adapter, border_trim)


	return gains_first, gains_head, gains_tail