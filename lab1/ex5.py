def concatenare_cuvinte(*args):
  rezultat = []
  for c in args:
    if type(c) == str:
        rezultat.append(c)
  return " ".join(rezultat)

print(concatenare_cuvinte("ana", "are", 2, "mere"))
print(concatenare_cuvinte("programarea", "calculatoarelor", "si", "limbaje", "de", "programare", 3))

"""sau
def concatenare_cuvinte(*args):
    return " ".join([c for c in args if isinstance(c, str)])
"""