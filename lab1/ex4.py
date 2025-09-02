from unidecode import unidecode

def transforma_cuvinte(s):
    txt = s.split(" ")
    rezultat = []
    for cuv in txt:
        rezultat.append( unidecode(cuv[0]).upper() + cuv[1:])
    return " ".join(rezultat)

print(transforma_cuvinte("Ana are mere"))
print(transforma_cuvinte("programarea calculatoarelor si limbaje de programare trei"))


""" sau

def transforma_cuvinte(s):
    return " ".join(unidecode(c[0]).upper() + c[1:] if c else "" for c in s.split())
"""