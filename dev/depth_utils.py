

def mat_to_str(matrix):
    s = ""
    for row in range(4):
        for col in range(4): 
            s = s + " " + str(matrix[row][col])
    return s.strip()