sir = input().replace(",", " ").replace(".", " ").split()

dic = {}

for i in sir:
    try:
        dic[i] += 1
    except KeyError:
        dic[i] = 1

for i,k in dic.items():
    print(f"Cuvantul '{i}' se regaseste de {k} ori")


"""sau:
import re

sir = input()
cuvinte = re.split(r"[,\.\s]+", sir.strip())

dic = {}

for cuv in cuvinte:
    if cuv:
        dic[cuv] = dic.get(cuv, 0) + 1

for cuv, frec in dic.items():
    print(f"Cuvantul '{cuv}' apare de {frec} ori.")

"""