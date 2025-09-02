def sort_tuple_list(l: list):
  for i,val in enumerate(l):
    for j,val in enumerate(l):
        if l[i][0] < l[j][0]:
            l[i], l[j] = l[j], l[i]
        else:
            if l[i][0] == l[j][0] and l[i][1] > l[j][1]:
              l[i], l[j] = l[j], l[i]
  return l

print(sort_tuple_list([(3, 1), (2, 2)]))  # => [(2, 2), (3, 1)]
print(sort_tuple_list([(0, 3), (3, 1), (2, 4), (10, 2), (1, 2), (0, 5)]))  # =>  [(0, 3), (0, 5), (1, 2), (2, 4), (3, 1), (10, 2)]
print(sort_tuple_list([(5, 6), (1, 3), (10, 5), (9, 1), (5, 7)]))  # =>  [(1, 3), (5, 6), (5, 7), (9, 1), (10, 5)]


"""sau

def sort_tuple_list(l: list):
    return sorted(l, key=lambda x: (x[0], -x[1]))

"""