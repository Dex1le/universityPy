def get_pins(observed_pin):
    adjacent_digits = {
        '1': ['1', '2', '4'],
        '2': ['1', '2', '3', '5'],
        '3': ['2', '3', '6'],
        '4': ['1', '4', '5', '7'],
        '5': ['2', '4', '5', '6', '8'],
        '6': ['3', '5', '6', '9'],
        '7': ['4', '7', '8'],
        '8': ['5', '7', '8', '9', '0'],
        '9': ['6', '8', '9'],
        '0': ['0', '8']
    }

    possible_combinations = ['']

    for digit in observed_pin:
        new_combinations = []
        for combination in possible_combinations:
            for adjacent_digit in adjacent_digits[digit]:
                new_combinations.append(combination + adjacent_digit)
        possible_combinations = new_combinations

    return possible_combinations

def main():
    observed_pin = input()
    result = get_pins(observed_pin)
    print(result)

if __name__ == "__main__":
    main()