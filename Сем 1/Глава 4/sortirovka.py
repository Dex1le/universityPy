def group_strings(strings):
    groups = {}
    for string in strings:
        key = ''.join(sorted(string))
        if key in groups:
            groups[key].append(string)
        else:
            groups[key] = [string]
    grouped_strings = list(groups.values())
    grouped_strings.sort(key=lambda x: len(x[0]))
    return grouped_strings

if __name__ == "__main__":
    strings = [string for string in input().split()]
    grouped_strings = group_strings(strings)
    for group in grouped_strings:
        print(group)