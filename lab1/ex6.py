def afisare_progresie_aritmetica(start, stop, pas = 2):
    [print(i) for i in range(start,stop,pas)]
      
afisare_progresie_aritmetica(1, 50, 5)
afisare_progresie_aritmetica(-5, 25)


"""sau mai eficient (nu creeaza lista)
def afisare_progresie_aritmetica(start, stop, pas=2): 
    list(map(print, range(start, stop, pas)))

"""