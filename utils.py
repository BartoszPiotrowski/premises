
def remove_supersets(list_of_sets):
    """Removes proper supersets from the list of sets"""
    list_of_sets_clean = []
    l = len(list_of_sets)
    for i1 in range(l):
        for i2 in range(l):
            if list_of_sets[i1] > list_of_sets[i2]:
                break
        else:
            list_of_sets_clean.append(list_of_sets[i1])
    return list_of_sets_clean
