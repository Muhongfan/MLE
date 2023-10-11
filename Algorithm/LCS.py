# The longest sub sequence
# Input: s1 = [1, 2, 3, 4, 1]

# def lss(nums):
#     n = len(nums)
#     if n<=2:
#         return 1
#     maxLen = [1]*n
#     for i in reversed(range(n)):
#         for j in range(i+1,n):
#             if nums[i]<nums[j]:
#                 maxLen[i] = max(maxLen[i], maxLen[j]+1)
#     print(maxLen)
#     return max(maxLen)
# print(lss([1,5,2,4,3]))

# The longest common sequence
# Input: s1 = [1, 2, 3, 4, 1], s2 = [3, 4, 1, 2, 1, 3]
# output: [1, 2, 3]

def longestCommonSubsequence(a, b):
    m = len(a)
    n = len(b)

    maxLen = [[0] * (n + 1) for _ in range(m + 1)]

    flag = 0
    p = 0
    for i in range(1, m):
        for j in range(1, n):
            if a[i - 1] == b[j - 1]:
                maxLen[i][j] = maxLen[i - 1][j - 1] + 1
                if flag < maxLen[i][j]:
                    flag = maxLen[i][j]
                    p = i
            else:
                maxLen[i][j] = max(maxLen[i - 1][j], maxLen[i][j - 1])
    #a[p-flag:p]
    return flag+1
s1 = "abcde"
s2 = "ace"
print(longestCommonSubsequence(s1, s2))

