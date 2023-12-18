def main():

    def long_str(s):

        char_map = {}
        start = 0
        max_length = 0

        for i in range(len(s)):
            if s[i] in char_map and start <= char_map[s[i]]:
                start = char_map[s[i]] + 1
            max_length = max(max_length, i - start + 1)
            char_map[s[i]] = i
        return max_length

    string = input()
    print(long_str(string))

if __name__ == '__main__':
    main()