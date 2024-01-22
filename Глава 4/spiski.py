def split_list(nums, C, N):
    total_sum = sum(nums)
    if total_sum % 2 != 0:
        return False

    target_sum = total_sum // 2
    current_sum = 0
    first_list = []
    second_list = []

    for num in nums:
        if current_sum + num <= target_sum:
            first_list.append(num)
            current_sum += num
        else:
            second_list.append(num)

    return True


if __name__ == '__main__':
     a = [2, 6, 10, 2]
     b = [1, 2, 8, 10]
     c = [2, 2, 4]
     d = [2, 3, 3, 3, 4, 5]
     print(split_list(a))
     print(split_list(b))
     print(split_list(c))
     print(split_list(d))