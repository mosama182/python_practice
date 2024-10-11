
# Just playing around with lamda function, map and filter


def test_lambda_func_single_arg(list1):

    y = list(map(lambda x: x**2, list1))

    print(y)

    return y


def test_lambda_func_two_arg(list1, list2):

    z = list(map(lambda x, y: x + y, list1, list2))

    print(z)


def filter_function_single_arg(list1):

    y = list(filter(lambda x: x < 5, list1))

    print(y)


if __name__ == "__main__":

    x = list(range(10))
    y = list(range(10, 20))

    #test_lambda_func_single_arg(x)

    #test_lambda_func_two_arg(x, y)

    filter_function_single_arg(x)