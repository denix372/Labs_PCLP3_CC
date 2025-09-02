def all_tuple_exist(l: list, element) -> bool:
    for tup in l:
        val = element in tup
        if val == False:
            return False
    return True

print(all_tuple_exist([(1, 4, 2, 3), (10, 11, 2, 3), (6, 7, 1)], 2)) # -> False
print(all_tuple_exist([("string", 2, -1), (-10, 5, "string", 1, -1.3), (2, "string", -0.5, 2)], "string")) # -> True

"""sau
def all_tuple_exist(l: list, element) -> bool:
    return all(element in tup for tup in l)
"""