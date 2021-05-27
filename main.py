import numpy
import networkx
import matplotlib.pyplot as plt
from networkx.algorithms.distance_measures import center

world_wide_web_matrix = numpy.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                     [0, 0, 1, 0, 1, 0, 1, 0, 1, 1],
                                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                     [0, 1, 0, 0, 1, 1, 1, 0, 0, 0],
                                     [0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
                                     [0, 1, 1, 1, 0, 0, 1, 0, 1, 1],
                                     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                     [0, 1, 0, 0, 0, 1, 1, 0, 1, 0],
                                     [1, 0, 1, 1, 1, 0, 0, 1, 0, 1],
                                     [0, 0, 1, 1, 1, 0, 1, 1, 0, 0]])


def draw_graph(adjacency_matrix: numpy.array, eigenvector: numpy.array) -> None:
    """
    Draws the graph 

    Input:
        adjacency_matrix (numpy.array): Adjacency matrix of the graph
        eigenvector (numpy.array): Eigenvector representing the rank for each page

    Output:
        None
    """
    G = networkx.from_numpy_matrix(
        adjacency_matrix, create_using=networkx.MultiDiGraph)
    shift = [0, -0.1]

    print("centro: ", center(G))

    pos = networkx.spring_layout(G)

    shifted_pos = {node: node_pos + shift for node, node_pos in pos.items()}

    networkx.draw(G, with_labels=True, pos=pos)

    labels = {}
    for page in range(adjacency_matrix.shape[0]):
        labels[page] = round(eigenvector[page, 0], 4)

    networkx.draw_networkx_labels(
        G, shifted_pos, labels=labels, horizontalalignment="left", font_size=8)

    # adjust frame to avoid cutting text, may need to adjust the value
    axis = plt.gca()
    axis.set_xlim([1.5*x for x in axis.get_xlim()])
    axis.set_ylim([1.5*y for y in axis.get_ylim()])

    plt.show()  # display


def get_probability_matrix(adjacency_matrix: numpy.array) -> numpy.single:
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

    print("# Outer links de cada pagina: ", page_outer_links_dict)

    return A


def page_rank(adjacency_matrix: numpy.array) -> numpy.single:
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

    #print("Probability Matrix:")
    # print(world_wide_web_matrix)

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

    while abs(error) > 0.00001:
        P_before_multiplication = P
        P = numpy.dot(world_wide_web_matrix, P_before_multiplication)
        error = numpy.sqrt(numpy.add(
            P.T, -1 * P_before_multiplication.T).dot(numpy.add(P, -1 * P_before_multiplication)))[0][0]

    rank_dict = {}

    # Assign rank to each page
    highest_page = 0

    for page in range(number_of_pages):
        page_rank = P[page, 0]
        rank_dict[page] = page_rank
        if page_rank > rank_dict[highest_page]:
            highest_page = page

    print("Highest page: ", highest_page)

    return P


eigen_vector = page_rank(world_wide_web_matrix)
print("eigen_vector:")
print(eigen_vector)
draw_graph(world_wide_web_matrix, eigen_vector)
