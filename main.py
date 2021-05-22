import numpy

world_wide_web_matrix = numpy.array([
    [0, 0, 1/2, 1/2],
    [1, 0, 0, 1/2],
    [0, 1/2, 0, 0],
    [0, 1/2, 1/2, 0]
])


def page_rank(world_wide_web_matrix: numpy.array):

    # Ensure the matrix is squared
    assert world_wide_web_matrix.shape[0] == world_wide_web_matrix.shape[1],\
        """page_rank Error: The world wide web matrix must be squared,
        rows: {} cols: {}.""".format(world_wide_web_matrix.shape[0],
                                     world_wide_web_matrix.shape[1])

    number_of_pages = world_wide_web_matrix.shape[0]

    # Create S column vector
    S = numpy.array([[1 / number_of_pages for i in range(number_of_pages)]]).T

    # Generate eigenvector

    P_before_multiplication = S
    P = numpy.dot(world_wide_web_matrix, P_before_multiplication)
    error = numpy.sqrt(numpy.add(
        P.T, -1 * P_before_multiplication.T).dot(numpy.add(P, -1 * P_before_multiplication)))[0][0]
    print(error)
    while abs(error) > 0.00001:
        P_before_multiplication = P
        P = numpy.dot(world_wide_web_matrix, P_before_multiplication)
        error = numpy.sqrt(numpy.add(
            P.T, -1 * P_before_multiplication.T).dot(numpy.add(P, -1 * P_before_multiplication)))[0][0]
        print(error)

    return P


print(world_wide_web_matrix)
eigen_vector = page_rank(world_wide_web_matrix)
print(eigen_vector)

# TODO
# - Get the the center or centers of the adjacency matrix
