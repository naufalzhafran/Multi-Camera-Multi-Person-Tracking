import numpy as np

dis_mat = np.array([
  [],[]
])

dis_mat[dis_mat<0.5] = 0

print(dis_mat)


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

assigment = linear_assignment(-dis_mat)

print(assigment)