def generate_permutations(nums):
    if len(nums) == 0:
        return [[]]

    permutations = []
    for i in range(len(nums)):
        remaining_nums = nums[:i] + nums[i+1:]
        for permutation in generate_permutations(remaining_nums):
            permutations.append([nums[i]] + permutation)

    return permutations

def main():
    nums = [int(num) for num in input().split()]
    permutations = generate_permutations(nums)
    for permutation in permutations:
        print(permutation)

if __name__ == "__main__":
    main()