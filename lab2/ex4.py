def all_characters_strings(l: list) -> set:
    s = set(l[0])

    for sir in l:
        s = s | set(sir)

    return s

print(all_characters_strings(["face", "include", "flat", "banner"])) # -> {a, b, c, d, e, f, i, l, n, r, t, u}