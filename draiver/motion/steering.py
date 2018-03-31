#!/envs/drAIver/bin/python


def find_intersection_point(line, intercept):
    if line is not None:
        return int(line[0] * (intercept ** 2) + line[1] * intercept + line[2])
    else:
        return None


def find_intersection_points(line_left, line_right, intercept):
    left_int = find_intersection_point(line_left, intercept)
    right_int = find_intersection_point(line_right, intercept)
    return left_int, right_int
