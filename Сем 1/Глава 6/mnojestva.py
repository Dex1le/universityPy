def get_subsets(elements):
    subsets = [[]]

    for element in elements:
        subsets += [subset + [element] for subset in subsets]

    return subsets

def count_unique_subsets(elements):
    unique_subsets = set(tuple(subset) for subset in get_subsets(elements) if len(subset) > 0)

    return len(unique_subsets), unique_subsets

def main():
    elements = list(set(element for element in input().split()))
    result_count, result_subsets = count_unique_subsets(elements)
    print(result_count)
    print(result_subsets)

if __name__ == "__main__":
    main()