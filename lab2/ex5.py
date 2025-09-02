def all_combinations_target(l: list, t: int):
    l = sorted(l)
    s = set(l) #mai eficient 
    rez = []
    for i in range(len(l) - 1):
        for j in range(i + 1, len(l)):
            k = t - l[i] - l[j]
            if k in s and k > l[j]:
                rez.append((l[i], l[j], k))
    return rez
      

print(all_combinations_target([1, 2, 3, 4, 5, 6, 7, 8, 9], 15))