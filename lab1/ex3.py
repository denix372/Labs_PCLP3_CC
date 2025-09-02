def numar_vocale(s):
  nr = 0
  for i in s:
    if i in "aeiou":
      nr += 1

  #sau nr = sum(1 for i in s if i in "aeiou")
  return nr

print(numar_vocale("ana are mere")) # => 6
print(numar_vocale("programarea este fun")) # => 8
print(numar_vocale("twyndyllyngs")) # => 0