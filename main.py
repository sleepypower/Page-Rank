import numpy

world_wide_web_matrix = numpy.array([
    [0, 0, 1/2, 1/2],
    [1, 0, 0, 0],
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
    S = numpy.array([[1/ number_of_pages for i in range(number_of_pages)]]).T



print(world_wide_web_matrix)
page_rank(world_wide_web_matrix)
