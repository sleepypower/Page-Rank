import numpy

world_wide_web_matrix = numpy.array([[0,   1,   0,   0,   0],
                                     [1,   0,   1,   1,   0],
                                     [1,   0,   0,   1,   1],
                                     [1,   1,   0,   0,   0],
                                     [0,   0,   1,   0,   0]])


def get_probability_matrix(adjacency_matrix: numpy.array):
    """
    Calculate each page outer links.
    In the adjacency matrix, a row represents a page, therefore each 1 in that 
    row represents an outer link for that page. 
    Calculate the number of '1s' for each page

    Input:
        adjacency_matrix (numpy.array): Adjacency matrix

    Output:
        A (numpy.single): Probability matrix
    """

    # Ensure the matrix is squared
    assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1],\
        """page_rank Error: The world wide web matrix must be squared,
        rows: {} cols: {}.""".format(adjacency_matrix.shape[0],
                                     adjacency_matrix.shape[1])

    number_of_pages = adjacency_matrix.shape[0]
    page_outer_links_dict = {}

    for row in range(number_of_pages):
        page_outer_links_dict[row] = 0
        for col in range(number_of_pages):
            if adjacency_matrix[row, col] == 1:
                page_outer_links_dict[row] += 1

    # Generate the Probability matrix A
    A = numpy.single([[0 for j in range(number_of_pages)]
                      for i in range(number_of_pages)])

    for row in range(number_of_pages):
        for col in range(number_of_pages):
            if adjacency_matrix[row, col] == 1:
                A[col, row] = (1 / page_outer_links_dict[row])

    print(page_outer_links_dict)

    return A


def page_rank(adjacency_matrix: numpy.array):
    """
    Calculates the rank for every page given an adjacency matrix
    Uses the formula lim n->Inf (A^n*S) = P where A is the adjacency matrix
    S is a vector containing 1/#Pages. Finally P is the eigenvector representing
    the rank of each page.

    Input:
        adjacency_matrix (numpy.array): Adjacency matrix of the world wide web
    Output:
        P (numpy.array): Eigenvector representing the rank of each page
    """
    # Get Probability matrix
    world_wide_web_matrix = get_probability_matrix(adjacency_matrix)

    print("Probability Matrix:")
    print(world_wide_web_matrix)

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
    # print(error)

    while abs(error) > 0.00001:
        P_before_multiplication = P
        P = numpy.dot(world_wide_web_matrix, P_before_multiplication)
        error = numpy.sqrt(numpy.add(
            P.T, -1 * P_before_multiplication.T).dot(numpy.add(P, -1 * P_before_multiplication)))[0][0]
        # print(error)

    return P


eigen_vector = page_rank(world_wide_web_matrix)
print(eigen_vector)

# TODO
# - Get the the center or centers of the adjacency matrix
# Como el Grafo es simple, solo puede haber un enlace de pagina B a la pagina C