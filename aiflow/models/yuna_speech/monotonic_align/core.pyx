cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void maximum_path_each(int[:,::1] path, float[:,::1] value, int t_y, int t_x, float max_neg_val=-1e9) noexcept nogil:
	cdef int x
	cdef int y
	cdef float v_prev
	cdef float v_cur
	cdef int index = t_x - 1
	if t_y < 1 or t_x < 1:
		return

	for y in range(t_y):
		for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
			if x == y:
				v_cur = max_neg_val
			else:
				v_cur = value[y - 1, x]
			if x == 0:
				v_prev = 0. if y == 0 else max_neg_val
			else:
				v_prev = value[y - 1, x - 1]
			value[y, x] += max(v_prev, v_cur)

	# y must stop at 1: the y == 0 lookback reads value[-1, :], which with
	# wraparound(False) is a raw read before the buffer (segfault on a fresh
	# per-item allocation). Its decrement is dead anyway — the loop ends.
	for y in range(t_y - 1, 0, -1):
		path[y, index] = 1
		if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
			index = index - 1
	path[0, index] = 1


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void maximum_path_c(int[:,:,::1] paths, float[:,:,::1] values, int[::1] t_ys, int[::1] t_xs) noexcept nogil:
	cdef int b = paths.shape[0]
	cdef int i
	for i in range(b):
		maximum_path_each(paths[i], values[i], t_ys[i], t_xs[i])
