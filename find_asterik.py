def find_asterik(data, start_point):

    M = len(data) # nos. of rows
    N = len(data[0]) # nos. of columns

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    if data[start_point[0]][start_point[1]] == '*':
        return start_point
    
    visited = []

    candidate_start_points = [start_point]

    while candidate_start_points:
        
        current_point = candidate_start_points.pop()

        if current_point in visited:
            continue

        visited.append(current_point)

        for move in moves:
            new_point = (current_point[0] + move[0], current_point[1] + move[1])
            if (new_point[0] <= M-1) and (new_point[1] <= N-1) and (new_point[0]>=0) and (new_point[1] >= 0):
                x = data[new_point[0]][new_point[1]]
                if x == '*':
                    return new_point
                elif x != '#':
                    candidate_start_points.append(new_point)
                else:
                    continue
    return None
        

if __name__ == '__main__':
    data = [[0, 0, 0, '*'], ['#', 0, 0, '#'], [0, 0, 0, 0], [0, 0, 0, 0]]
    start_point = (1, 2)

    print(find_asterik(data, start_point))


