num = [1,1,2,2,3,4,5,5,5,6]

n = len(num) 

def remove_duplicates(nums):
    if n == 0:
        return 0
    
    j = 0
    for i in range(1, n):
        if nums[i] != nums[j]:
            j += 1
            nums[j] = nums[i]
    
    return j + 1

x = remove_duplicates(num)
print("The number of unique elements is:", x)
print("The modified list is:", num[:x])

