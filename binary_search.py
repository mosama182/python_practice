def binary_search(array, target):

    left = 0
    right = len(array) - 1

    while left <= right:

        mid = (left + right) // 2

        if array[mid] == target:
            return mid
        
        elif target < array[mid]:
            right = mid - 1

        elif target > array[mid]:
            left = mid + 1

    
    return -1


def majority_element(array, left, right):

    if left == right:
        return array[left]
    
    mid = (left + right) // 2

    left_majority = majority_element(array, left, mid)
    right_majority = majority_element(array, mid + 1, right)

    if left_majority == right_majority:
        return left_majority
    else:
        left_count = count_occurence(array, left, mid, left_majority)
        right_count = count_occurence(array, mid + 1, right, right_majority)

        if left_count > right_count:
            return left_majority
        else:
            return right_majority


def count_occurence(array, left, right, majority):
    """
    Count the occurence of the element 'majority'
    in the given array between indices left and right
    """
    
    return array[left:right+1].count(majority)


def two_sum(array, target):

    seen = {}

    for i, elem in enumerate(array):
        diff = target - elem
        if diff in seen:
            return [seen[diff], i]
        else:
            seen[elem] = i

    return []

    
if __name__ == "__main__":

    # =========== #
    # binary search
    # =========== #
    #array = [0, 1, 2, 3, 4, 5]
    #target = -1
    #print(binary_search(array, target))

    # ============== #
    # majority element
    # ============== #
    #array = [2, 2, 3, 3, 3]
    #print(majority_element(array, 0, len(array)-1))

    # ====== #
    # two sum
    # ====== #
    array = [1, 2, 3]
    target = 3
    print(two_sum(array, target))

