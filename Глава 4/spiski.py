def find_closest_sum(nums, target):
    nums.sort()  # Сначала отсортируем список чисел

    closest_sum = float('inf')  # Инициализируем переменную с бесконечным значением
    result = []

    for i in range(len(nums) - 2):
        left = i + 1
        right = len(nums) - 1

        while left < right:
            current_sum = nums[i] + nums[left] + nums[right]
            if abs(current_sum - target) < abs(closest_sum - target):
                closest_sum = current_sum
                result = [nums[i], nums[left], nums[right]]

            if current_sum < target:
                left += 1
            elif current_sum > target:
                right -= 1
            else:
                return result

    return result

def main():
    nums = [int(num) for num in input().split()]
    target = int(input())
    closest_sum = find_closest_sum(nums, target)
    print(closest_sum)

if __name__ == "__main__":
    main()
