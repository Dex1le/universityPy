def brackets_right(s):
    stack = []
    matching = {'(': ')', '[': ']', '{': '}'}
    for char in s:
        if char in matching.keys():
            stack.append(char)
        elif char in matching.values():
            if not stack or matching[stack.pop()] != char:
                return False
    return len(stack) == 0


def LongestRightS(s):
    if brackets_right(s):
        return True
    else:
        res = ""
        n = 0
        for l in range(len(s) - 1):
            for r in range(l + 1, len(s)):
                if brackets_right(s[l:r]):
                    if n < len(s[l:r]):
                        n = len(s[l:r])
                        res = s[l:r]
        if res:
            return res
        else:
            return False


def main():
    s = input()
    print(LongestRightS(s))


if __name__ == "__main__":
    main()