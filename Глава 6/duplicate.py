def compare_lists(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1.intersection(set2)
    unique_to_list1 = set1.difference(set2)
    unique_to_list2 = set2.difference(set1)
    remaining_in_list1 = len(set1.difference(intersection))
    remaining_in_list2 = len(set2.difference(intersection))

    return len(intersection), len(unique_to_list1), remaining_in_list1, remaining_in_list2

list1 = [0, 33, 37, 6, 10, 44, 13, 47, 16, 18, 22, 25]
list2 = [1, 38, 48, 8, 41, 7, 12, 47, 16, 40, 20, 23, 25]

result = compare_lists(list1, list2)
print(result)