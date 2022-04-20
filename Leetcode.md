***
1. 两数之和
    * 构建hash表
    * 遍历数组，每次判断target-num是否在hash表中
    * 如果不在就将当前num以及index填入hash表
    * 如果在就输出当前num和hash表中num的index
```python
from typing import List
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hash_table = dict()
        for idx, num in enumerate(nums):
            if target - num not in hash_table:
                hash_table[num] = idx
            else:
                return [hash_table[target - num], idx]
```

***
2. 两数相加
     * 使用l1、l2的val构建head（用于返回）
     * cur指向当前head（用于计算）
     * 当l1和l2的next存在时，则向后推，否则为空节点
     * 首先构建cur.next，然后更新当前cur（注意进位）
     * cur向后推
     * 结束时处理当前cur大于10的情况
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        head = ListNode(l1.val + l2.val)
        cur = head

        while l1.next or l2.next:
            l1 = l1.next if l1.next else ListNode()
            l2 = l2.next if l2.next else ListNode()

            cur.next = ListNode(l1.val + l2.val + cur.val // 10)
            cur.val = cur.val % 10

            cur = cur.next
        
        if cur.val >= 10:
           cur.next = ListNode(cur.val // 10)
           cur.val = cur.val % 10
        return head
```

***
3. 无重复字符的最长字串
    * 构建指针rk，ans记录结果长度，occ为当前维护的不重复字符集合
    * 遍历字符串，除了首位以外，每次从occ中删除字符串中的上一个字符（因为已经计算完以该字符起始的最大长度了）
    * 持续移动指针rk，如果s[rk]不在occ中，将其加入进去（rk为以当前i起始的最大连续不重复字符串的结尾）
    * 使用rk-i更新ans
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        n = len(s)
        res = 0
        ptr = 0
        visited = set()

        for i in range(n):
            if i != 0:
                visited.remove(s[i - 1])
                
            while ptr < n and s[ptr] not in visited:
                visited.add(s[ptr])
                ptr += 1
            res = max(res, ptr - i)

        return res
```

***
4. 寻找两个正序数组的中位数
    * 保持第一个数组长度不大于第二个数组（否则直接交换）
    * 计算共同中位数左侧数量
    * 计算nums1的中位数m1 = left + (right - left) // 2，并获取nums2中位数k-m1
    * 如果nums1中位数右侧数大于nums2中位数左侧，搜索m1右侧，否则搜索m1左侧
    * m1为left，m2为k-m1
    * c1为nums1、nums2中位数左的大值（注意边界条件0），基数直接返回c1
    * 偶数计算c2，中位数右侧的小值
```python
from typing import List
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        n1 = len(nums1)
        n2 = len(nums2)

        if n1 > n2:
            return self.findMedianSortedArrays(nums2, nums1)

        left, right = 0, n1
        k = (n1 + n2 + 1) // 2
        while left < right:
            mid1 = left + (right - left) // 2
            mid2 = k - mid1

            if nums1[mid1] < nums2[mid2 - 1]:
                left = mid1 + 1
            else:
                right = mid1

        mid1 = left
        mid2 = k - mid1
        c1 = max(nums1[mid1 - 1] if mid1 > 0 else float('-inf'), nums2[mid2 - 1] if mid2 > 0 else float('-inf'))
        if (n1 + n2) % 2 == 1:
            return c1
        c2 = min(nums1[mid1] if mid1 < n1 else float('inf'), nums2[mid2] if mid2 < n2 else float('inf'))
        return (c1 + c2) / 2
```

***
5. 最长回文子串
    * 构建dp表，表示i -> j是否为回文子串
    * 遍历回文长度（由小到大的长度，方便转移方程）
    * 遍历首位字符索引i，计算末位字符索引j（如果j>=n结束）
    * 如果l==0则为T
    * 如果l==1且字符ij相同则为T
    * 如果l>=2，则判断dp[i + 1][j - 1]，以及当前ij索引字符是否相同
    * 如果dp[i][j]为T且长度大于当前最优，则更新最长字符串
```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        res = ''

        for l in range(n):
            for i in range(n):
                j = i + l
                if j >= n:
                    break

                if l == 0:
                    dp[i][j] = True
                elif l == 1:
                    dp[i][j] = (s[i] == s[j])
                else:
                    dp[i][j] = dp[i + 1][j - 1] and (s[i] == s[j])

                if dp[i][j] and l + 1 > len(res):
                    res = s[i: j + 1]
        return res
```

***
10. 正则表达式匹配
    * 构建dp表，存放s中前i个字符是否与p中前j个匹配
    * 初始化dp[0][0]=T
    * 遍历，其中由于p为空时，s任何值都为F，所有p从1开始遍历
    * 如果p中元素为*，则判断上一个字符与s中当前字符是否匹配（其中.一定为T，s为空时一定不匹配）
    * 如果匹配，则返回s中去掉该字符的匹配结果与p中去掉该正则组合的匹配结果的|
    * 如果不匹配，则返回p中去掉该正则组合的匹配结果
    * 如果p中元素不为*，即为字符或.，则判断s与p中当前字符是否匹配
    * 如果匹配，则为dp[i - 1][j - 1]的匹配结果
    * 如果不匹配，则为F
    * 匹配中，如果没遇到*，那p有字符，s为空时一定不匹配
    * 如果遇到*，那上一个字符与空串s匹配直接返回F，跳转到比p中去掉这个组合的结果（如果为空串则为T）
```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m = len(s)
        n = len(p)

        def match(i, j):
            if i == 0:
                return False
            if p[j - 1] == '.':
                return True
            return s[i - 1] == p[j - 1]

        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        for i in range(m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    if match(i, j - 1):
                        dp[i][j] = dp[i - 1][j] | dp[i][j - 2]
                    else:
                        dp[i][j] = dp[i][j - 2]
                else:
                    if match(i, j):
                        dp[i][j] = dp[i - 1][j - 1]
        return dp[m][n]
```

***
11. 盛水最多容器
    * 构建左右指针
    * 每次计算当前max_area和左右指针之间的area中较大的
    * 移动低的一侧指针
```python
from typing import List
class Solution:
    def maxArea(self, height: List[int]) -> int:
        left, right = 0, len(height) - 1
        area = 0
        while left < right:
            area = max(area, min(height[left], height[right]) * (right - left))
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return area
```

***
15. 三数之和
    * 先进行排序
    * 遍历第一个数字（如果数字大于0且与上一个重复，则直接跳过）
    * 计算二三数字的和，将第三个数字设为末位，并从first+1遍历第二个数字（如果数字大于0且与上一个重复，则直接跳过）
    * 遍历第三个数字寻找指针位置
```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        res = []
        for first in range(n):
            if first > 0 and nums[first] == nums[first - 1]:
                continue
            third = n - 1
            target = - nums[first]
            for second in range(first + 1, n):
                if second > first + 1 and nums[second] == nums[second - 1]:
                    continue
                while second < third and nums[second] + nums[third] > target:
                    third -= 1
                
                if second == third:
                    break
                
                if nums[second] + nums[third] == target:
                    res.append([nums[first], nums[second], nums[third]])
        return res
```

***
17. 电话号码的字母组合
    * 构建map
    * 回溯如果长度等于n则添加
    * 否则查找该数字对应字母
    * 遍历字母进行添加，回溯，pop
```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []

        PhoneMap = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz",
        }

        def dfs(idx):
            if idx == len(digits):
                combinations.append("".join(combination))
            else:
                digit = PhoneMap[digits[idx]]
                for ch in digit:
                    combination.append(ch)
                    dfs(idx + 1)
                    combination.pop()

        combination = []
        combinations = []
        dfs(0)
        return combinations
```

***
19. 删除链表的倒数第 N 个结点
    * 定义dummy用于返回next
    * 构建first、second双指针
    * first先走n步
    * 双指针共同前进到first抵达结尾
    * 取second.next = second.next.next（跳过待删除节点）
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        dummy = ListNode(0, head)
        fast = head
        slow = dummy
        for _ in range(n):
            fast = fast.next
        
        while fast:
            fast = fast.next
            slow = slow.next
            
        slow.next = slow.next.next
        return dummy.next
```

***
20. 有效的括号
    * 构建括号对，以右括号为键
    * 构建栈，并遍历字符串
    * 如果为左括号直接入栈
    * 如果为右括号，判断当前栈为空或栈顶不是对应左括号，则为F，否则消掉栈顶对应括号
```python
class Solution:
    def isValid(self, s: str) -> bool:
        if len(s) % 2 == 1:
            return False

        pairs = {
            ")": "(",
            "}": "{",
            "]": "["
        }

        stack = []
        for ch in s:
            if ch in pairs:
                if not stack or stack[-1] != pairs[ch]:
                    return False
                stack.pop()
            else:
                stack.append(ch)
        return not stack
```

***
21. 合并两个有序链表
    * 如果l1或者l2任何一个为None，返回另一个（递归终止条件）
    * 判断l1与l2的val大小，返回较小的一个（因为是上一个节点连接过来）
    * 将next与另一个节点再进行merge（期待返回里面较小的一个）
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1:
            return l2
        if not l2:
            return l1
        
        if l1.val <= l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2
```

***
22. 括号生成
    * 定义ans来存放结果
    * 定义左括号和右括号数量进行回溯，最大均为n，定义当前排序s
    * 首先判断总长度为2n则直接填入ans
    * 如果左括号小于n，则添加左括号、左括号数量+1回溯、pop
    * 如果右括号数量小于左括号数量，则添加右括号、右括号数量+1回溯，pop（保持右括号左侧有大于自己数量的左括号）
```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res = []

        def dfs(s, left, right):
            if len(s) == 2 * n:
                res.append("".join(s))
                return
            if left < n:
                s.append('(')
                dfs(s, left + 1, right)
                s.pop()
            if right < left:
                s.append(')')
                dfs(s, left, right + 1)
                s.pop()

        dfs([], 0, 0)
        return res
```

***
23. 合并K个升序链表
    * 左右指针，自上而下分解大lists，自下而上归并
    * 归并时，对于左右list，进行升序归并
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        if not lists:
            return []
        n = len(lists)
        return self.merge(lists, 0, n - 1)

    def merge(self, lists, left, right):
        if left == right:
            return lists[left]
        mid = left + (right - left) // 2
        l1 = self.merge(lists, left, mid)
        l2 = self.merge(lists, mid + 1, right)
        return self.mergeTwoList(l1, l2)

    def mergeTwoList(self, l1, l2):
        if not l1:
            return l2
        if not l2:
            return l1
        if l1.val <= l2.val:
            l1.next = self.mergeTwoList(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoList(l1, l2.next)
            return l2
```

***
31. 下一个排列
    * 从右向左查找第一个小于自身右侧数字的较小数（尽可能靠右）
    * 如果存在，则再次从右向左找第一个比这个数大的数（较大数尽可能小）
    * 交换这两个数（右侧一定为降序）
    * 原始排列中较小数所在索引，右侧所有数字升序排列（此时右侧一定为降序，左右逐个交换即可，保证为下一个，即排列尽可能小）
```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        i = len(nums) - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        
        if i >= 0:
            j = len(nums) - 1
            while j >= 0 and nums[j] <= nums[i]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
        
        left, right = i + 1, len(nums) - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
```

***
32. 最长有效括号
    * 定义栈，存储索引，用于计算长度
    * 遍历，如果栈为空、当前字符为左括号、栈顶为右括号，则直接入栈
    * 否则，说明栈顶为左括号，当前为右括号，组成了一个合法括号
    * 栈顶出栈，更新最大长度，如果栈为空，则只减去首位索引-1
```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        stack = []
        n = len(s)
        res = 0
        for i in range(n):
            if not stack or s[i] == '(' or s[stack[-1]] == ')':
                stack.append(i)
            else:
                stack.pop()
                res = max(res, i - (stack[-1] if stack else -1))
        return res
```

***
33. 搜索旋转排序数组
    * 如果没有数字返回-1
    * 左右指针l,r
    * 取中位数mid，如果mid为target，直接返回
    * 如果首位<=中位数，则左侧有序，判断target是否在首位中位之间
    * 否则，右侧有序，判断target是否在中位末位之间
```python
from typing import List
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if not nums:
            return -1
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] == target:
                return mid

            if nums[0] <= nums[mid]:
                if nums[0] <= target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            else:
                if nums[mid] < target <= nums[-1]:
                    l = mid + 1
                else:
                    r = mid - 1
        return -1
```

***
33. 搜索旋转排序数组
    * 如果没有数字返回-1
    * 左右指针l,r
    * 取中位数mid，如果mid为target，直接返回
    * 如果首位<=中位数，则左侧有序，判断target是否在首位中位之间
    * 否则，右侧有序，判断target是否在中位末位之间
```python
from typing import List
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if not nums:
            return -1
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] == target:
                return mid

            if nums[0] <= nums[mid]:
                if nums[0] <= target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            else:
                if nums[mid] < target <= nums[-1]:
                    l = mid + 1
                else:
                    r = mid - 1
        return -1
```

***
34. 在排序数组中查找元素的第一个和最后一个位置
    * 如果没有nums或者target最右侧数，则返回[-1, -1]
    * 否则返回[左侧二分查找结果, 右侧二分查找结果]
    * 左侧二分查找，如果mid为target，则right = mid - 1（寻找最左侧数，令left为target）
    * 右侧二分查找，如果mid为target，则left = mid + 1（寻找最右侧数，令right为target）
```python
from typing import List
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if not nums or target > nums[-1]:
            return [-1, -1]
        return [self._search_left(nums, target), self._search_right(nums, target)]

    def _search_left(self, nums, target):
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] > target:
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1
            elif nums[mid] == target:
                right = mid - 1  # 改动这里，使搜索区间向左侧收缩
        if left >= len(nums) or nums[left] != target:  # 判断索引越界情况
            return -1
        return left

    def _search_right(self, nums, target):
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] > target:
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1
            elif nums[mid] == target:
                left = mid + 1  # 改动这里，使搜索区间向右侧收缩
        if right < 0 or nums[right] != target:  # 判断索引越界情况
            return -1
        return right
```

***
39. 组合总和
    * 对数组进行排序
    * 定义ans记录结果
    * 记录位置、当前组合、剩余量
    * 对当前位置到结尾进行遍历（相当于遍历当前位置，当前组合情况下的所有满足条件情况）
    * 如果当前位置数等于剩余量，将[当前组合, 当前数字]加入ans中（终止条件）
    * 如果小于剩余量，从当前位置向后（数字可重复），更新当前组合和剩余量
```python
from typing import List
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()

        ans = []
        def dfs(s, use, remain):
            for i in range(s, len(candidates)):
                c = candidates[i]
                if c == remain:
                    ans.append(use + [c])

                if c < remain:
                    dfs(i, use + [c], remain - c)

                if c > remain:
                    return
        dfs(0, [], target)
        return ans
```

***
42. 接雨水
    * 构建leftmax和rightmax，存放每个索引左右最大高度
    * 遍历更新leftmax、rightmax
    * 遍历计算每个索引对应的leftmax和rightmax中较小的一个与自身高度之差作为接水量
```python
from typing import List
class Solution:
    def trap(self, height: List[int]) -> int:
        if not height:
            return 0

        n = len(height)
        leftmax = [height[0]] + [0] * (n - 1)
        for i in range(1, n):
            leftmax[i] = max(leftmax[i - 1], height[i])

        rightmax = [0] * (n - 1) + [height[n - 1]]
        for i in range(n - 2, -1, -1):
            rightmax[i] = max(rightmax[i + 1], height[i])

        ans = sum(min(leftmax[i], rightmax[i]) - height[i] for i in range(n))
        return ans
```

***
46. 全排列
    * 维护需要遍历的子数组第一个位置索引first
    * 如果为n，则填入ans
    * 遍历first到n，每次与first交换、回溯first+1、复原（相当于遍历所有组合方式）
```python
from typing import List
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def backtrack(first):
            if first == n:
                ans.append(nums[:])
            for i in range(first, n):
                nums[first], nums[i] = nums[i], nums[first]
                backtrack(first + 1)
                nums[i], nums[first] = nums[first], nums[i]

        n = len(nums)
        ans = []
        backtrack(0)
        return ans
```

***
48. 旋转图像
    * 水平翻转（行）：[i, j] [n - 1 - i, j]
    * 对角线翻转：[i, j] [j, i]
```python
from typing import List
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        n = len(matrix)
        for i in range(n // 2):
            for j in range(n):
                matrix[i][j], matrix[n - i - 1][j] = matrix[n - i - 1][j], matrix[i][j]
        for i in range(n):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
```

***
49. 字母异位词分组
    * 构建hash表
    * 遍历数组，将每次字符串组合作为值，字符串作为键加入其中
```python
from typing import List
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        mp = {}.fromkeys([])
        for st in strs:
            key = "".join(sorted(st))
            mp[key].append(st)
        return list(mp.values())
```

***
53. 最大子序和
    * 构建最大和结果res，当前最大结果tmp_sum
    * 遍历数组，更新tmp_sum，如果为负数则直接取当前数，否则加上当前数
```python
from typing import List
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        tmp_sum = 0
        res = nums[0]
        for num in nums:
            tmp_sum = max(tmp_sum + num, num)
            res = max(res, tmp_sum)
        return res
```

***
55. 跳跃游戏
    * 构建向右可达到最远位置rightmost
    * 遍历数组，如果当前位置可达到，即i <= rightmost，更新rightmo
    * 如果rightmost达到数组长度，则为T
```python
from typing import List
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n, rightmost = len(nums), 0
        for i in range(n):
            if i <= rightmost:
                rightmost = max(rightmost, i + nums[i])
                if rightmost >= n - 1:
                    return True
        return False
```

***
56. 合并区间
    * 按起始索引排序
    * 构建merged存方最终结果
    * 遍历，判断merged为空或当前最后list的末位索引小于下一list的首位索引，则添加到merged
    * 否则合并
```python
from typing import List
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])

        merged = []
        for interval in intervals:
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                merged[-1][1] = max(merged[-1][1], interval[1])
        return merged
```

***
62. 不同路径
    * 构建m*n的dp表，第一行第一列为1，表示均只有一种方法到达
    * 遍历，由上和左加和得到
```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[1] * n] + [[1] + [0] * (n - 1) for _ in range(m - 1)]
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = f[i - 1][j] + f[i][j - 1]
        return dp[m - 1][n - 1]
```

***
64. 最小路径和
    * 构建dp表，第一行第一列直接顺次加总
    * 遍历，每次左和上中较小的一个加上当前位置上数值
```python
from typing import List
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0
        m, n = len(grid), len(grid[0])
        dp = [[0] * n for _ in range(m)]
        dp[0][0] = grid[0][0]

        for i in range(1, m):
            dp[i][0] = dp[i - 1][0] + grid[i][0]
        for j in range(1, n):
            dp[0][j] = dp[0][j - 1] + grid[0][j]

        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
        return dp[m - 1][n - 1]
```

***
70. 爬楼梯
    * 构建a，b存放上一阶和两阶的方法数
```python
class Solution:
    def climbStairs(self, n: int) -> int:
        a = b = 1
        for i in range(2, n + 1):
            a, b = b, a + b
        return b
```

***
72. 编辑距离
    * 构建dp表，表示从A的前i个到B的前j个的编辑距离
    * 任何一个为空字符串，返回另一个的长度
    * 初始化任何一个空字符串，dp为另一个的长度
    * 每次上个A串插入一个（当前A删掉一个变成上个A，再转成目标B）
    * 上个B串插入一个（当前A转成上个B，再插入一个变成当前B）
    * 上个A上个B的A新加入的换成B新加入的（此时需要考虑新加入是否相同）
    * 取三种情况最小，更新dp
```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)

        if m * n == 0:
            return m + n

        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                left = dp[i - 1][j] + 1
                up = dp[i][j - 1] + 1
                left_up = dp[i - 1][j - 1]
                if word1[i - 1] != word2[j - 1]:
                    left_up += 1
                dp[i][j] = min(left, up, left_up)
        return dp[m][n]
```

***
75. 颜色分类
    * 构建首尾指针
    * 遍历，每次遇到2则反复与尾指针交换（保证不会把尾部的2交换过来）
    * 遇到0与首指针交换
```python
from typing import List
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        n = len(nums)
        p0, p2 = 0, n - 1
        i = 0
        while i <= p2:
            while i <= p2 and nums[i] == 2:
                nums[i], nums[p2] = nums[p2], nums[i]
                p2 -= 1
            if nums[i] == 0:
                nums[i], nums[p0] = nums[p0], nums[i]
                p0 += 1
            i += 1
```

***
76. 最小覆盖子串
    * 使用目标字符串t初始化哈希表need
    * needCnt记录目标字符串中所有字符数量
    * res存放首尾索引结果
    * 初始化i为后指针
    * 遍历数组，如果哈希表中该字符大于0，则needCnt - 1（说明找到了一个目标字符）
    * 哈希表中该字符数量-1（目标字符趋于0，多于字符为负数）
    * 如果needCnt为0，表示前指针已经找到了符合条件的窗口，准备移动后指针
    * 不断移动后指针，每次哈希表中对应字符数量+1，直到找到哈希表中对应值为0的元素（即第一个目标字符）
    * 确定后指针后，更新res，更新哈希表（哈希表中将该字符对应数量+1）、needCnt、后指针位置
    * 返回结果为前后指针之间区间
```python
import collections
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        need = dict()
        for c in t:
            if c not in need:
                need[c] = 1
            else:
                need[c] += 1
        
        needCnt = len(t)
        res = (0, float('inf'))
        i = 0

        for j, c in enumerate(s):
            if c in need and need[c] > 0:
                needCnt -= 1
            if c not in need:
                need[c] = -1
            else:
                need[c] -= 1
            
            if needCnt == 0:
                while True:
                    c_move = s[i]
                    if need[c_move] == 0:
                        break
                    need[c_move] += 1
                    i += 1
                
                if j - i < res[1] - res[0]:
                    res = (i, j)
                
                need[s[i]] += 1
                needCnt += 1
                i += 1
        return '' if res[1] >= len(s) else s[res[0]: res[1] + 1]
```

***
78. 子集
    * 遍历，每次将res中所有list加上当前数，添加到res中
```python
from typing import List
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = [[]]
        for i in nums:
            res = res + [[i] + li for li in res]
        return res
```

***
79. 单词搜索
    * 构建搜索方向数组
    * visited存放以当前起始位置，所搜索过的位置
    * 遍历每一个位置，是否满足条件
    * 如果当前位置不等于单词中对应字母，则F
    * 如果k达到单词长度，则T
    * 将当前位置填入visited
    * 搜索当前位置的四个方向，如果新位置满足条件，且不在visited中
    * 判断当前位置与单词的后续字母
    * 将该遍历位置移出visited，开始新位置的遍历
```python
from typing import List
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        def check(i: int, j: int, k: int) -> bool:
            if board[i][j] != word[k]:
                return False
            if k == len(word) - 1:
                return True

            visited.add((i, j))
            result = False
            for di, dj in directions:
                newi, newj = i + di, j + dj
                if 0 <= newi < len(board) and 0 <= newj < len(board[0]):
                    if (newi, newj) not in visited:
                        if check(newi, newj, k + 1):
                            result = True
                            break

            visited.remove((i, j))
            return result

        h, w = len(board), len(board[0])
        visited = set()
        for i in range(h):
            for j in range(w):
                if check(i, j, 0):
                    return True

        return False
```

***
84. 柱状图中最大的矩形
    * 构建左右list，记录每个idx对应左右最近的低于当前数字的索引，用于记录以当前高度为高的最大面积
    * 定义单调栈
    * 遍历，如果栈内有值，且栈顶高度大于当前高度，则更新right中栈顶位置索引，栈顶出栈（left只需要考虑左侧小于当前元素的索引，因为有更小的当前值，所以左侧大于的值直接出栈，且对于出栈的位置来说，当前值即为其右侧第一个小值）
    * 若栈顶高度不大于当前值，则更新left为栈顶值（空栈则为-1），当前值入栈
    * 计算结果，每个索引值与其left、right计算最大值
```python
from typing import List
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        n = len(heights)
        if n == 0:
            return 0
        left, right = [0] * n, [n] * n

        stack = list()
        for i in range(n):
            while stack and heights[stack[-1]] >= heights[i]:
                right[stack[-1]] = i
                stack.pop()
            left[i] = stack[-1] if stack else -1
            stack.append(i)

        ans = max((right[i] - left[i] - 1) * heights[i] for i in range(n))
        return ans
```

***
85. 最大矩形
    * 构建前缀和pre，用于计算当前行之上的柱形情况
    * 遍历每行，更新前缀和，遇到0则直接置为0，1则+1
    * 定义单调栈
    * 遍历前缀和柱形
    * 如果栈不为空，且栈顶元素大于当前元素，则将栈顶元素弹出
    * 计算以栈顶元素为高，栈中下一个元素（左边第一个小于栈顶元素的idx），以及当前元素（右侧第一个小于栈顶元素的idx），为底的矩形面积，更新res
    * 当前元素入栈
```python
from typing import List
class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        if not matrix:
            return 0
        m, n = len(matrix), len(matrix[0])

        pre = [0] * (n + 1)
        res = 0

        for i in range(m):
            for j in range(n):
                pre[j] = pre[j] + 1 if matrix[i][j] == '1' else 0

            stack = [-1]
            for idx, num in enumerate(pre):
                while stack and pre[stack[-1]] > num:
                    k = stack.pop()
                    res = max(res, pre[k] * (idx - stack[-1] - 1))
                stack.append(idx)
        return res
```

***
94. 二叉树的中序遍历
    * 递归，如果没有根节点则return（达到叶子节点）
    * 左，append，右
```python
from typing import List
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        def dfs(root):
            if not root:
                return
            dfs(root.left)
            res.append(root.val)
            dfs(root.right)

        dfs(root)
        return res
```

***
96. 不同的二叉搜索树
    * G(n)：长度为n的序列能构成的不同二叉搜索树个数
    * F(i, n)：以i为根、序列长度为n的不同二叉搜索树个数
    * G(n) = sum(F(i, n)) G(0) = G(1) = 1
    * 将F(i, n)根据i拆分为左右子树的乘积，构建G(i-1)、G(n-i)
```python
class Solution:
    def numTrees(self, n: int) -> int:
        G = [0] * (n + 1)
        G[0], G[1] = 1, 1
        for i in range(2, n + 1):
            for j in range(1, i + 1):
                G[i] += G[j - 1] * G[i - j]
        return G[n]
```

***
98. 验证二叉搜索树
    * 自上而下遍历，定义upper和lower
    * 如果到达叶子节点，直接返回T
    * 取当前节点值val
    * 如果val <= lower或者val >= upper则返回False
    * 如果判别左侧或右侧失败，则F
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:

        def discriminator(node, lower, upper):
            if not node:
                return True
            val = node.val
            if val <= lower or val >= upper:
                return False

            return discriminator(node.left, lower, val) and discriminator(node.right, val, upper)
        return discriminator(root, float('-inf'), float('inf'))
```

***
101. 对称二叉树

    * 如果没有根节点，则直接返回T
    * 递归左右子节点
    * 如果左右子节点都没有，则为T
    * 如果一个有一个没有，则为F
    * 如果都有，则比较值
    * 返回左左比右右，左右比右左
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True
        def dfs(left, right):
            if not left and not right:
                return True
            if not left or not right:
                return False
            if left.val != right.val:
                return False
            return dfs(left.left, right.right) and dfs(left.right, right.left)
        return dfs(root.left, root.right)
```

***
102. 二叉树的层序遍历

    * 如果没有根节点，返回[]
    * res储存结果
    * 构建队列que，root入队
    * 遍历队列（队列不断更新）
    * 遍历队列长度（tmp存放该层所有节点值）
    * 元素出队，并存放其值（遍历结束后，该层所有节点出队）
    * 将其左右子节点入队，作为下一层的遍历对象
```python
from typing import List
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []

        res = []
        que = [root]
        while que:
            size = len(que)
            tmp = []
            for _ in range(size):
                first = que.pop(0)
                tmp.append(first.val)
                if first.left:
                    que.append(first.left)
                if first.right:
                    que.append(first.right)
            res.append(tmp)
        return res
```

***
104. 二叉树的最大深度

    * 如果没有根节点，返回0（递归终止条件）
    * 递归左侧最大深度，右侧最大深度
    * 两侧最大深度中的max + 1（根节点）
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        left_height = self.maxDepth(root.left)
        right_height = self.maxDepth(root.right)
        return max(left_height, right_height) + 1
```

***
105. 从前序与中序遍历序列构造二叉树

    * 构建字典存储节点与对应中序idx
    * 根据前序中序的左右索引构建二叉树
    * 如果前序的左索引大于右索引，则返回None
    * 前序根节点索引为前序左（前序：根左右）
    * 使用前序中根节点索引，在idx字典中获取中序根节点索引
    * 构建根节点
    * 根据中序列表中根节点索引计算左侧子树长度
    * 调整前序中序的左右指针，构建左右子树
```python
from typing import List
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        def BuildTree(preorder_l, preorder_r, inorder_l, inorder_r):
            if preorder_l > preorder_r:
                return None

            preorder_root = preorder_l
            inorder_root = index[preorder[preorder_root]]

            root = TreeNode(preorder[preorder_root])
            subtree_l_s = inorder_root - inorder_l

            root.left = BuildTree(preorder_l + 1, preorder_l + subtree_l_s, inorder_l, inorder_root - 1)
            root.right = BuildTree(preorder_l + subtree_l_s + 1, preorder_r, inorder_root + 1, inorder_r)

            return root

        n = len(preorder)
        index = {element: i for i, element in enumerate(inorder)}
        return BuildTree(0, n - 1, 0, n - 1)
```

***
114. 二叉树展开为链表

    * 前序遍历记录list
    * 遍历list，顺序连接
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        preorder = list()
        def preorderTraversal(root):
            if root:
                preorder.append(root)
                preorderTraversal(root.left)
                preorderTraversal(root.right)

        preorderTraversal(root)
        size = len(preorder)
        for i in range(1, size):
            prev, cur = preorder[i - 1], preorder[i]
            prev.left = None
            prev.right = cur
```

***
121. 买卖股票的最佳时机

    * 维护历史最低价，总最大利润
    * 遍历，每次更新历史最低价
    * 使用当前价格与历史最低价差更新最大利润
```python
from typing import List
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        minprice = float('inf')
        maxprofit = 0
        for price in prices:
            minprice = min(minprice, price)
            maxprofit = max(maxprofit, price - minprice)
        return maxprofit
```

***
124. 二叉树中的最大路径和

    * 构建全局maxSum用于存放结果
    * 递归遍历，自下而上每次计算每个节点的最大贡献值，即以该节点为根节点起点，在其子树中的最大路径和
    * 空节点贡献值为0，非空节点贡献值为当前节点值+左右节点中较大的一个
    * 使用当前节点值与其左右节点最大贡献值来更新maxSum，返回贡献值
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def __init__(self):
        self.maxSum = int('-inf')

    def maxPathSum(self, root: TreeNode) -> int:
        def maxGain(node):
            if not node:
                return 0

            leftGain = max(maxGain(node.left), 0)
            rightGain = max(maxGain(node.right), 0)

            maxres = node.val + leftGain + rightGain

            self.maxSum = max(self.maxSum, maxres)

            return node.val + max(leftGain, rightGain)

        maxGain(root)
        return self.maxSum
```

***
128. 最长连续序列

    * 维护最长连续序列长度longest_streak
    * 遍历数组，遇到num - 1不在集合中才进行新一轮操作（否则长度一定不如num - 1的才操作）
    * 取当前num进行操作，更新连续序列长度以及最大连续序列长度
```python
from typing import List
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        longest_streak = 0
        num_set = set(nums)

        for num in num_set:
            if num - 1 not in num_set:
                current_num = num
                current_streak = 1

                while current_num + 1 in num_set:
                    current_num += 1
                    current_streak += 1

                longest_streak = max(longest_streak, current_streak)
        return longest_streak
```

***
136. 只出现一次的数字

    * 位运算
```python
from typing import List
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        res = nums[0]
        for i in range(1, len(nums)):
            res = res ^ nums[i]
        return res
```

***
139. 单词拆分

    * 构建dp表，表示每个位置是否可由给出word列表到达
    * 遍历首尾索引，判断当前首索引是否可到达，构成字符串是否在列表中
    * 更新尾索引处dp表
```python
from typing import List
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True
        for i in range(n):
            for j in range(i + 1, n + 1):
                if dp[i] and (s[i: j] in wordDict):
                    dp[j] = True
        return dp[-1]
```

***
141. 环形链表

    * 构建集合存储走过的节点
    * 遍历每步判断是否出现过
```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        if not head or not head.next:
            return False
        
        slow = head
        fast = head.next

        while slow != fast:
            if not fast or not fast.next:
                return False
            slow = slow.next
            fast = fast.next.next
        
        return True
```

***
142. 环形链表 II

    * 构建快慢指针
    * 如果没有环，则返回
    * 快慢指针相遇处break（此时分别走了nb和2nb步）
    * 快指针置于起始位置，返回快慢指针相遇位置（s = a + nb，让快指针走未知的a步与满指针相遇）
```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        fast, slow = head, head
        while True:
            if not fast or not fast.next:
                return
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                break
        fast = head
        while fast != slow:
            fast = fast.next
            slow = slow.next
        return fast
```

***
146. LRU 缓存机制

    * hash_table + 双向链表
```python
class DLinkedNode:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = dict()
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.capacity = capacity
        self.size = 0

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self.moveToHead(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key not in self.cache:
            node = DLinkedNode(key, value)
            self.cache[key] = node
            self.addToHead(node)
            self.size += 1
            if self.size > self.capacity:
                removed = self.removeTail()
                self.cache.pop(removed.key)
                self.size -= 1
        else:
            node = self.cache[key]
            node.value = value
            self.moveToHead(node)

    def addToHead(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def removeNode(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def moveToHead(self, node):
        self.removeNode(node)
        self.addToHead(node)

    def removeTail(self):
        node = self.tail.prev
        self.removeNode(node)
        return node
```

***
148. 排序链表

    * 对首尾指针进行排序
    * 如果没有head，则返回
    * 如果head与tail相连，将head指向None（切断两部分），返回head
    * 定义快慢指针寻找mid
    * 对head至mid，以及mid至tail进行归并（自上而下递归完成全部cut操作）
    * 遍历两节点，每次按大小合并，返回dummy.next（自下而上逐步完成merge操作）
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        def sortFunc(head, tail):
            if not head:
                return None
            if head.next == tail:
                head.next = None
                return head
            slow = fast = head
            while fast != tail:
                slow = slow.next
                fast = fast.next
                if fast != tail:
                    fast = fast.next
            mid = slow
            return merge(sortFunc(head, mid), sortFunc(mid, tail))

        def merge(head1, head2):
            dummy = ListNode(0)
            tmp, tmp1, tmp2 = dummy, head1, head2
            while tmp1 and tmp2:
                if tmp1.val <= tmp2.val:
                    tmp.next = tmp1
                    tmp1 = tmp1.next
                else:
                    tmp.next = tmp2
                    tmp2 = tmp2.next
                tmp = tmp.next
            if tmp1:
                tmp.next = tmp1
            elif tmp2:
                tmp.next = tmp2
            return dummy.next
        return sortFunc(head, None)
```

***
152. 乘积最大子数组

    * 每次更新前i个的最小值，最大值
    * 使用最大值与当前值、最小值与当前值、当前值（考虑当前值为正负两种情况）
```python
from typing import List
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        if not nums:
            return
        res = nums[0]
        pre_max = nums[0]
        pre_min = nums[0]

        for num in nums[1:]:
            cur_max = max(pre_max * num, pre_min * num, num)
            cur_min = min(pre_max * num, pre_min * num, num)
            res = max(res, cur_max)
            pre_max = cur_max
            pre_min = cur_min
        return res
```

***
155. 最小栈

    * 栈
```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = [float("inf")]

    def push(self, x):
        self.stack.append(x)
        self.min_stack.append(min(x, self.min_stack[-1]))

    def pop(self):
        self.stack.pop()
        self.min_stack.pop()

    def top(self):
        return self.stack[-1]

    def getMin(self):
        return self.min_stack[-1]
```

***
160. 相交链表

    * 定义双指针，任何一个先到达终点，立刻指向另一个指针起点
    * 两指针相交时，恰好为相交点
```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        A, B = headA, headB
        while A != B:
            A = A.next if A else headB
            B = B.next if B else headA
        return A
```

***
169. 多数元素

    * 构建hash表，存放每个数字出现数量
    * 遍历计算
```python
from typing import List
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        counts = dict()
        for i in nums:
            if i not in counts:
                counts[i] = 1
            else:
                counts[i] += 1
        return max(counts.keys(), key=counts.get)
```

***
198. 打家劫舍

    * 维护两个连续时刻能获取的最大利益
    * 遍历，更新最大利益
```python
from typing import List
class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums:
            return 0

        size = len(nums)
        if size == 1:
            return nums[0]

        first, second = nums[0], max(nums[0], nums[1])
        for i in range(2, size):
            first, second = second, max(first + nums[i], second)
        return second
```

***
200. 岛屿数量

    * 遍历每个位置，如果为1，则数量+1，开始搜索周围连接
    * 首先将搜索位置置为0，防止重复搜索
    * 遍历上下左右，如果满足条件，则继续搜索
```python
from typing import List
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        def dfs(grid, r, c):
            grid[r][c] = '0'
            nr, nc = len(grid), len(grid[0])
            for x, y in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
                if 0 <= x < nr and 0 <= y < nc and grid[x][y] == '1':
                    dfs(grid, x, y)

        nr = len(grid)
        if nr == 0:
            return 0
        nc = len(grid[0])

        num_islands = 0
        for r in range(nr):
            for c in range(nc):
                if grid[r][c] == '1':
                    num_islands += 1
                    dfs(grid, r, c)
        return num_islands
```

***
206. 反转链表

    * 如果next为None，则返回
    * 递归（递归到倒数第二个节点）
    * head.next.next = head（下一个节点的下一个节点为当前节点，相当于翻转）
    * 下一个节点为None
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if head == None or head.next == None:
            return head
        cur = self.reverseList(head.next)
        head.next.next = head
        head.next = None
        return cur
```

***
207. 课程表

    * 定义邻接表，定义状态向量
    * 遍历课程表，将每个后续课程填入前缀课程对应list中（每个列表为该课程所延展出的课程）
    * 遍历课程，将每个课程视为初始节点
    * 递归当前课程
    * 如果当前课程状态为-1，则已被访问过，返回T
    * 如果当前课程状态为1，则正被其他节点访问，存在环，返回F
    * 如果当前课程状态为0，则从未被访问，将其状态置为1，表示正被访问，随后遍历其对应后续课程
    * 遍历完成后，将其状态设为-1
```python
from typing import List
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        def dfs(i, adj, flags):
            if flags[i] == -1:
                return True
            if flags[i] == 1:
                return False
            flags[i] = 1
            for j in adj[i]:
                if not dfs(j, adj, flags):
                    return False
            flags[i] = -1
            return True

        adj = [[] for _ in range(numCourses)]
        flags = [0 for _ in range(numCourses)]
        for cur, pre in prerequisites:
            adj[pre].append(cur)
        for i in range(numCourses):
            if not dfs(i, adj, flags):
                return False
        return True
```

***
208. 实现 Trie (前缀树)
```python
class Trie:
    def __init__(self):
        self.children = [None] * 26
        self.isEnd = False

    def searchPrefix(self, prefix):
        node = self
        for ch in prefix:
            ch = ord(ch) - ord('a')
            if not node.children[ch]:
                return None
            node = node.children[ch]
        return node

    def insert(self, word):
        node = self
        for ch in word:
            ch = ord(ch) - ord('a')
            if not node.children[ch]:
                node.children[ch] = Trie()
            node = node.children[ch]
        node.isEnd = True

    def search(self, word):
        node = self.searchPrefix(word)
        return node is not None and node.isEnd

    def startsWith(self, prefix):
        return self.searchPrefix(prefix) is not None
```

***
215. 数组中的第K个最大元素

    * 记录target为对应排序后的位置
    * 进行快速排序获取索引
    * 比较该索引与target大小，进行调整
```python
from typing import List
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        size = len(nums)
        target = size - k

        left = 0
        right = size - 1
        while True:
            index = self.__partition(nums, left, right)
            if index == target:
                return nums[index]
            elif index < target:
                left = index + 1
            else:
                right = index - 1

    def __partition(self, nums, left, right):
        import random
        random_index = random.randint(left, right)
        nums[random_index], nums[left] = nums[left], nums[random_index]

        pivot = nums[left]
        while left < right:
            while left < right and nums[right] >= pivot:
                right -= 1
            nums[left] = nums[right]
            while left < right and nums[left] <= pivot:
                left += 1
            nums[right] = nums[left]
        nums[left] = pivot
        return left
```

***
221. 最大正方形

    * 定义maxSide存放最大边长
    * 构建dp表，存放以ij为右下角的最大正方形面积
    * 遍历，如果首行或首列，则直接为1
    * 取左、左上、上三者中最小值加一更新dp表
    * 更新maxSide
```python
from typing import List
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return 0
        maxSide = 0
        r, c = len(matrix), len(matrix[0])
        dp = [[0] * c for _ in range(r)]
        for i in range(r):
            for j in range(c):
                if matrix[i][j] == '1':
                    if i == 0 or j == 0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                    maxSide = max(maxSide, dp[i][j])
        maxSquare = maxSide ** 2
        return maxSquare
```

***
226. 翻转二叉树

    * 递归至叶子节点
    * 翻转叶子节点
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return root

        left = self.invertTree(root.left)
        right = self.invertTree(root.right)
        root.left, root.right = right, left
        return root
```

***
234. 回文链表

    * 定义头指针
    * 递归至尾节点（如果当前节点不存在返回T终止）
    * 逐步与头指针比较（如果下一个节点有问题则返回F）
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        self.front = head

        def cur_check(cur_node):
            if cur_node is not None:
                if not cur_check(cur_node.next):
                    return False
                if self.front.val != cur_node.val:
                    return False
                self.front = self.front.next
            return True
        return cur_check(head)
```

***
236. 二叉树的最近公共祖先

    * 如果没有root（抵达叶子节点）或者root为p或q（找到p或q提取结束），则直接返回root
    * 递归左侧，递归右侧
    * 左右同时为空，则该子树不包含pq，返回
    * 均不为空，则该节点为最近公共祖先，直接返回
    * 一侧空一侧不空，此时该侧为p、q之一，或该侧即为最近公共祖先（下面包括另一节点）
```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root or root == p or root == q:
            return root

        l = self.lowestCommonAncestor(root.left, p, q)
        r = self.lowestCommonAncestor(root.right, p, q)
        if not l and not r:
            return
        if not l:
            return r
        if not r:
            return l
        return root
```

***
238. 除自身以外数组的乘积

    * ans存放结果
    * 构建从左至右以及从右至左的累乘L，R（忽略首位和末位）
    * 遍历更新ans
```python
from typing import List
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        l = len(nums)
        L, R, ans = [0] * l, [0] * l, [0] * l

        L[0] = 1
        R[0] = 1
        for i in range(1, l):
            L[i] = L[i - 1] * nums[i - 1]
            R[i] = R[i - 1] * nums[l - i]
        for i in range(l):
            ans[i] = L[i] * R[l - i - 1]
        return ans
```

***
239. 滑动窗口最大值

    * 构建单调队列q（只需保证当前窗口最大的在队列中，且队列降序，因为左侧较小的之后不会用到）
    * 将第一个窗口逐个加入队列，保持队列降序（如果当前元素大于等于队尾，则将队尾逐个弹出）
    * 维护ans存放结果
    * 后续窗口逐个入队，保持队列降序
    * 每次窗口移动时，如果队首索引不在窗口内，则将队首弹出
    * 使用队首元素更新ans
```python
import collections
from typing import List
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        q = collections.deque()
        for i in range(k):
            while q and nums[i] >= nums[q[-1]]:
                q.pop()
            q.append(i)

        ans = [nums[q[0]]]
        for i in range(k, n):
            while q and nums[i] >= nums[q[-1]]:
                q.pop()
            q.append(i)

            while q[0] <= i - k:
                q.popleft()
            ans.append(nums[q[0]])
        return ans
```

***
240. 搜索二维矩阵 II

    * 从左下角向右上角寻找
    * 如果大于target，行-1
    * 如果小于target，列+1
```python
from typing import List
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return False
        h = len(matrix)
        w = len(matrix[0])

        r = h - 1
        c = 0

        while c < w and r >= 0:
            if matrix[r][c] > target:
                r -= 1
            elif matrix[r][c] < target:
                c += 1
            else:
                return True
        return False
```

***
279. 完全平方数

    * 构建候选平方数
    * 构建dp表，代表加和为某个数的最小平方数个数
    * 遍历dp表
    * 遍历平方数，如果大于当前idx，则break
    * 更新dp表
```python
class Solution:
    def numSquares(self, n: int) -> int:
        square_nums = [i ** 2 for i in range(1, n + 1) if i ** 2 <= n]
        dp = [float("inf")] * (n + 1)
        dp[0] = 0
        for i in range(1, n + 1):
            for square in square_nums:
                if i < square:
                    break
                dp[i] = min(dp[i], dp[i - square] + 1)
        return dp[-1]
```

***
283. 移动零

    * 双指针，左指针为已经处理好的序列尾部（右指针将0换过来左指针才移动），右指针为待处理序列头部（右指针一直移动）
    * 移动r指针寻找非0数，并与l指针交换
```python
from typing import List
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        l = r = 0
        while r < n:
            if nums[r] != 0:
                nums[l], nums[r] = nums[r], nums[l]
                l += 1
            r += 1
```

***
287. 寻找重复数

    * 左右指针，取数字范围的二分索引（1到n，n为数组长度-1）
    * 统计数组中严格小于等于二分索引的个数
    * 如果大于二分索引，根据抽屉原理，说明要找数字在左侧，否则在右侧
```python
from typing import List
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        size = len(nums)
        l = 1
        r = size - 1

        while l < r:
            mid = l + (r - l) // 2
            cnt = 0
            for num in nums:
                if num <= mid:
                    cnt += 1
            if cnt > mid:
                r = mid
            else:
                l = mid + 1
        return l
```

***
297. 二叉树的序列化与反序列化

    * 序列化时，遇到空节点，返回null
    * 递归，根节点 + , + 左子节点 + , + 右子节点，完成序列化（获取类似前序遍历结果）
    * 反序列化时，先将字符串转成list
    * 递归，每次出队节点为root，在按顺序还原左右节点，返回root
```python
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Codec:
    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return 'None'
        return str(root.val) + ',' + str(self.serialize(root.left)) + ',' + str(self.serialize(root.right))

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        def dfs(datalist):
            val = datalist.pop(0)
            if val == 'None':
                return None

            root = TreeNode(int(val))
            root.left = dfs(datalist)
            root.right = dfs(datalist)
            return root

        datalist = data.split(',')
        return dfs(datalist)
```

***
300. 最长递增子序列

    * 构建dp表，表示以第i个数为结尾的最长长度，且i必须被选中
    * 遍历，每次用前面dp中末位数字小于当前数字中长度最大的+1来更新dp
```python
from typing import List
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums:
            return 0
        n = len(nums)
        dp = [1] * n
        for i in range(n):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)
```

***
301. 删除无效的括号

    * 从原始字符串开始向下逐层遍历，每层为上一层元素删除一个元素的所有可能情况（只考虑左右括号，不考虑其他字符）
    * 每层判断是否有含有合理括号的情况，如果有则直接返回（保证删除最小的）
    * 判断是否有效括号，构建cnt计数，遇到左括号+1，右括号-1，如果小于0或最后不为0则为F
```python
from typing import List
class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        def isVaild(s):
            cnt = 0
            for c in s:
                if c == "(":
                    cnt += 1
                elif c == ")":
                    cnt -= 1
                if cnt < 0:
                    return False
            return cnt == 0

        level = {s}
        while True:
            valid = list(filter(isVaild, level))
            if valid:
                return valid

            next_level = set()
            for item in level:
                for i in range(len(item)):
                    if item[i] in '()':
                        next_level.add(item[: i] + item[i + 1:])

            level = next_level
```

***
309. 最佳买卖股票时机含冷冻期

    * 维护当日持有一只股票、不持有但处于冷冻期（当天执行卖出操作）、不持有且不处于冷冻期，三种状态
    * 遍历更新三种状态
    * f0 = max(f0, f2 - p)
    * f1 = f0 + p
    * f2 = max(f1, f2)
    * 返回f1和f2中最大的一个（f0持有股票未变现）
```python
from typing import List
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0
        n = len(prices)

        f0, f1, f2 = -prices[0], 0, 0
        for i in range(1, n):
            newf0 = max(f0, f2 - prices[i])
            newf1 = f0 + prices[i]
            newf2 = max(f1, f2)
            f0, f1, f2 = newf0, newf1, newf2

        return max(f1, f2)
```

***
312. 戳气球

    * 左右扩展1，构建dp表，表示(i, j)开区间能获取的最大金币数
    * 遍历区间长度2到n，遍历起始位置，更新dp表
    * 维护取区间内任意元素为mid时的最大值，取dp[i][k]、dp[k][j]为左右最大值，加上当前mid与ij的最大值
    * 更新dp表
```python
from typing import List
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        nums = [1] + nums + [1]
        n = len(nums)
        dp = [[0] * n for _ in range(n)]

        def rangemax(i, j):
            val_max = 0
            for k in range(i + 1, j):
                left = dp[i][k]
                right = dp[k][j]
                val = left + nums[i] * nums[k] * nums[j] + right
                if val > val_max:
                    val_max = val
            dp[i][j] = val_max

        for span in range(2, n):
            for i in range(0, n - span):
                rangemax(i, i + span)
        return dp[0][n - 1]
```

***
322. 零钱兑换

    * 构建dp表，表示凑成该索引钱数所需最小硬币数
    * 遍历所有数值，遍历所有硬币种类数
    * 更新dp表
```python
from typing import List
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0

        for x in range(1, amount + 1):
            for coin in coins:
                if x < coin:
                    continue
                dp[x] = min(dp[x], dp[x - coin] + 1)
            break
        return dp[-1] if dp[-1] != float("inf") else -1
```

***
337. 打家劫舍 III

    * 对每个节点设置偷和不偷两种状态
    * 递归遍历，自下而上更新状态
    * 节点rob，则左右节点不被rob时最大值与该节点值相加
    * 节点不被rob，则左右节点rob或不rob最大值相加
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def rob(self, root: TreeNode) -> int:
        def post_traverse(node):
            if not node:
                return [0, 0]
            left = post_traverse(node.left)
            right = post_traverse(node.right)
            do_rob = left[1] + right[1] + node.val
            no_rob = max(left[0], left[1]) + max(right[0], right[1])
            return [do_rob, no_rob]
        out = post_traverse(root)
        return max(out[0], out[1])
```

***
338. 比特位计数

    * 取小于该数，且最接近的2的整数次幂作为高位基准比特数（z & (z - 1) = 0）
    * 此时bit[x] = bit[y] + bit[z]，x = y + z，z为高位基准比特数，bit[z]=1
    * 构建bits用于存放结果，highBit维护当前高位基准比特数
    * 遍历，每次先更新高位基准比特数，再计算当前比特数，bits[i - highBit] + 1
```python
from typing import List
class Solution:
    def countBits(self, n: int) -> List[int]:
        bits = [0]
        highBit = 0
        for i in range(1, n + 1):
            if i & (i - 1) == 0:
                highBit = i
            bits.append(bits[i - highBit] + 1)
        return bits
```

***
347. 前 K 个高频元素

    * count统计后排序
```python
from typing import List
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        from collections import Counter
        count = Counter(nums)
        return [item[0] for item in count.most_common(k)]
```

***
394. 字符串解码

    * 构造辅助栈，遍历每个字符c
    * c为数字时，更新multi
    * c为字母时，在当前res尾部添加c
    * c为[时，将multi（[]内字符串重复次数）与res（相当于前面的字符串）入栈，并重置
    * c为]时，stack出栈，拼接字符串res = last_res + cur_multi * res（last_res为[]前面记录的字符串）
```python
class Solution:
    def decodeString(self, s: str) -> str:
        stack, res, multi = [], '', 0
        for c in s:
            if c == '[':
                stack.append([multi, res])
                res, multi = '', 0
            elif c == ']':
                cur_multi, last_res = stack.pop()
                res = last_res + cur_multi * res
            elif '0' <= c <= '9':
                multi = multi * 10 + int(c)
            else:
                res += c
        return res
```

***
399. 除法求值

    * 用字典构建图，节点之间权重（反向为倒数）
    * 集合存放节点
    * 遍历所有节点组合，遍历所有中间节点，如果连接存在，则将未连接节点连接起来
    * 遍历，如果节点见连接存在，则填入结果list
```python
from typing import List
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        graph = dict()
        set_node = set()
        for i in range(len(equations)):
            a, b = equations[i]
            graph[(a, b)] = values[i]
            graph[(b, a)] = 1 / values[i]
            set_node.add(a)
            set_node.add(b)

        arr = list(set_node)
        for k in arr:
            for i in arr:
                for j in arr:
                    if graph.get((i, k), None) and graph.get((k, j), None):
                        graph[(i, j)] = graph[(i, k)] * graph[(k, j)]

        res = []
        for x, y in queries:
            if graph.get((x, y), None):
                res.append(graph[(x, y)])
            else:
                res.append(-1)
        return res
```

***
406. 根据身高重建队列

    * 每个人按身高从高到低排序（身高相同按前面排得人升序，先排前面排得人少的）
    * 排序后顺序处理，后面的人不影响当前人排序，所以只需要插队即可
    * 将每个人插入到他前面有的人的位置
```python
from typing import List
class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        people.sort(key=lambda x: (-x[0], x[1]))
        ans = list()
        for p in people:
            ans[p[1]:p[1]] = [p]
        return ans
```

***
416. 分割等和子集

    * 如果不足两个数字，返回F
    * 如果总和为奇数，返回F
    * 如果最大数大于总合的一半，返回F
    * 构建dp表，存放从0到i的数能否构成j
    * j为0时不需要数字，故全为T
    * i为0时只能选择第一个数字，故j也为该数才为T，dp[0][nums[0]] = True
    * 遍历每个数字num
    * 遍历目标数，如果j < num，则直接取上一个状态dp[i][j] = dp[i - 1][j]
    * 如果j >= num，则如果选取该num，dp[i - 1][j - num]，如果不选取dp[i - 1][j]，两者取或
```python
from typing import List
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        n = len(nums)
        if n < 2:
            return False

        total = sum(nums)
        maxNum = max(nums)
        if total % 2 == 1:
            return False

        target = total // 2
        if maxNum > target:
            return False

        dp = [[False] * (target + 1) for _ in range(n)]
        for i in range(n):
            dp[i][0] = True

        dp[0][nums[0]] = True
        for i in range(1, n):
            num = nums[i]
            for j in range(1, target + 1):
                if j >= num:
                    dp[i][j] = dp[i - 1][j] | dp[i - 1][j - num]
                else:
                    dp[i][j] = dp[i - 1][j]
        return dp[n - 1][target]
```

***
437. 路径总和 III

    * 自顶向下，每次记录当前层加和种类list
    * 每次根据当前数字更新list（list种每个数字+当前数字，当前数字）
    * 统计当前层中加和等于目标值的数量
    * 返回当前count与左右节点的和
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> int:
        def dfs(root, sumlist):
            if root is None:
                return 0

            sumlist = [root.val + num for num in sumlist] + [root.val]

            count = 0
            for num in sumlist:
                if num == targetSum:
                    count += 1
            return count + dfs(root.left, sumlist) + dfs(root.right, sumlist)
        return dfs(root, [])
```

***
438. 找到字符串中所有字母异位词

    * 构建数组记录每个字母出现次数（如果字符串长度小于目标串长度则无）
    * 遍历前m个字母初始化s_cnt（过程中变化）和p_cnt（过程中不变）
    * 遍历后续字符，每次s_cnt中减少首个字符增加新字符
    * 每步判断s_cnt与p_cnt是否相等，如果相等则返回首位索引
```python
from typing import List
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        n, m = len(s), len(p)
        res = []

        if n < m:
            return res

        p_cnt = [0] * 26
        s_cnt = [0] * 26

        for i in range(m):
            p_cnt[ord(p[i]) - ord('a')] += 1
            s_cnt[ord(s[i]) - ord('a')] += 1

        if s_cnt == p_cnt:
            res.append(0)

        for i in range(m, n):
            s_cnt[ord(s[i - m]) - ord('a')] -= 1
            s_cnt[ord(s[i]) - ord('a')] += 1
            if s_cnt == p_cnt:
                res.append(i - m + 1)

        return res
```

***
448. 找到所有数组中消失的数字

    * 遍历，每次将该数对应的索引+n（由于可能加过n，所以需要进行取余操作）
    * 结束后，所有不大于n的位置索引即为结果
```python
from typing import List
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        n = len(nums)
        for num in nums:
            x = (num - 1) % n
            nums[x] += n
        ret = [i + 1 for i, num in enumerate(nums) if num <= n]
        return ret
```

***
461. 汉明距离

    * 异或操作
    * bin转为二进制
    * 统计1的数量即为汉明距离
```python
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        return bin(x ^ y).count('1')
```

***
494. 目标和

    * 如果target大于数组总和，则F
    * 设正数总和x，负数总和y，则x = (target + nums_sum) // 2，如果x不是整数则F
    * 构建dp表，存放从0到i的数能否构成j
    * j为0时不需要数字，故全为T
    * 遍历每一个数字num
    * 遍历目标数，如果j < num，则直接取上一个状态dp[i][j] = dp[i - 1][j]
    * 如果j >= num，则如果选取该num，dp[i - 1][j - num]，如果不选取dp[i - 1][j]，两者加和
```python
from typing import List
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        nums_sum = sum(nums)
        if target > nums_sum or (target + nums_sum) % 2 != 0:
            return 0
        x = (target + nums_sum) // 2

        n = len(nums)
        dp = [[0] * (x + 1) for _ in range(n)]

        for i in range(n):
            dp[i][0] = 1

        for i in range(n):
            num = nums[i]
            for j in range(x + 1):
                if j >= num:
                    dp[i][j] = dp[i - 1][j] + dp[i - 1][j - num]
                else:
                    dp[i][j] = dp[i - 1][j]
        return dp[n - 1][x]
```

***
538. 把二叉搜索树转换为累加树

    * 使用total作为全局中间变量，记录从小到大的累计和
    * 使用反中序遍历（右中左），利用total修改每个结点的值
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def convertBST(self, root: TreeNode) -> TreeNode:
        self.total = 0

        def dfs(root):
            if root:
                dfs(root.right)
                self.total += root.val
                root.val = self.total
                dfs(root.left)

        dfs(root)
        return root
```

***
543. 二叉树的直径

    * 递归遍历二叉树
    * 维护全局最大节点数ans，每次用左右子树最大深度+1来更新ans
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        self.ans = 1
        def dfs(root):
            if not root:
                return 0

            L = dfs(root.left)
            R = dfs(root.right)

            self.ans = max(self.ans, L + R + 1)

            return max(L, R) + 1
        dfs(root)
        return self.ans - 1
```

***
560. 和为K的子数组

    * 构建哈希表（前序累加和：该累加和出现次数）
    * res存放总出现次数，cur_sum记录当前累加和
    * 遍历数组，更新cur_sum，如果cur_sum - k存在于哈希表中，证明二者直接序列为可行序列，存入res
    * 更新哈希表
```python
from typing import List
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        numsum_times = {0: 1}
        cur_sum = 0
        res = 0

        for num in nums:
            cur_sum += num

            if cur_sum - k in numsum_times:
                res += numsum_times[cur_sum - k]

            numsum_times[cur_sum] = numsum_times.get(cur_sum, 0) + 1
        return res
```

***
581. 最短无序连续子数组

    * 初始化右指针为0，从左向右寻找最大值（保证指针指向第一个不更新最大值，即需要调整的数）
    * 如果num大于等于当前最大值，则更新最大值，否则更新右指针
    * 初始化左指针为n - 1，从右向左寻找最小值（保证指针指向第一个不更新最小值，即需要调整的数）
    * 如果num小于等于当前最小值，则更新最小值，否则更新左指针
```python
from typing import List
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        numsum_times = {0: 1}
        cur_sum = 0
        res = 0

        for num in nums:
            cur_sum += num

            if cur_sum - k in numsum_times:
                res += numsum_times[cur_sum - k]

            numsum_times[cur_sum] = numsum_times.get(cur_sum, 0) + 1
        return res
```

***
617. 合并二叉树

    * 同时遍历两棵树
    * 如果都没有节点则返回None
    * 如果其中一个有，一个没有，返回有的那个
    * 如果都有，返回两者之和
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def mergeTrees(self, root1: TreeNode, root2: TreeNode) -> TreeNode:
        if not root1:
            return root2
        if not root2:
            return root1

        merged = TreeNode(root1.val + root2.val)
        merged.left = self.mergeTrees(root1.left, root2.left)
        merged.right = self.mergeTrees(root1.right, root2.right)
        return merged
```

***
621. 任务调度器

    * 统计出现次数最多的任务
    * （最多出现次数 - 1） * （冷却时间 + 1） + 出现次数等于最多次数的任务数
    * 如果任务数大于上述结果，则只需要插入排列即可，返回总任务数
```python
from typing import List
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        from collections import Counter
        freq = Counter(tasks)
        max_exec = max(freq.values())

        max_tasks = sum([1 for v in freq.values() if v == max_exec])

        return max((max_exec - 1) * (n + 1) + max_tasks, len(tasks))
```

***
647. 回文子串

    * 遍历回文串中心（一个字符或者连续两个字符）
    * 统计每个中心向外扩散可以获得的回文子串数
    * 将所有中心的结果汇总
```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        def extend(s, i, j, n):
            res = 0
            while i >= 0 and j < n and s[i] == s[j]:
                i -= 1
                j += 1
                res += 1
            return res

        res = 0
        n = len(s)
        for i in range(n):
            res += extend(s, i, i, n)
            res += extend(s, i, i + 1, n)
        return res
```

***
739. 每日温度

    * 构建单调栈，维护索引
    * 遍历数组，每次判断栈内有数字且当前数字大于栈顶数字
    * 当前数字即为栈顶数字的下一个大数，将栈顶索引弹出，并更新ans
    * 直到栈顶为空或当前数字不再大于栈顶，则将当前数字入栈
```python
from typing import List
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        ans = [0] * n
        stack = []
        for i in range(n):
            temperature = temperatures[i]
            while stack and temperature > temperatures[stack[-1]]:
                prev_idx = stack.pop()
                ans[prev_idx] = i - prev_idx
            stack.append(i)
        return ans
```

***
***
剑指 Offer 03. 数组中重复的数字

    * 构建哈希表，用于存放数组中元素
    * 遍历数组，如果在哈希表中存在，则直接返回，否则加入哈希表
```python
from typing import List
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        hash_table = set()
        for num in nums:
            if num in hash_table:
                return num
            else:
                hash_table.add(num)
        return -1
```

***
剑指 Offer 04. 二维数组中的查找

    * 从左下角开始找
    * 如果当前元素大于target，向上移动一行
    * 如果当前元素小于target，向右移动一列
    * 如果等于则返回T，遍历完未找到返回F
```python
from typing import List
class Solution:
    def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
        i, j = len(matrix) - 1, 0
        while i >= 0 and j < len(matrix[0]):
            if matrix[i][j] > target:
                i -= 1
            elif matrix[i][j] < target:
                j += 1
            else:
                return True
        return False
```

***
剑指 Offer 05. 替换空格

    * 构建list存放结果
    * 遍历字符串，遇到空格则将%20加入list中，否则将字符加入到list中
    * 最后将list转为字符串
```python
class Solution:
    def replaceSpace(self, s: str) -> str:
        res = []
        for c in s:
            if c == ' ':
                res.append('%20')
            else:
                res.append(c)
        return ''.join(res)
```

***
剑指 Offer 06. 从尾到头打印链表

    * 递归链表，每次传入head.next，如果head为空则返回[]
    * 每次将上一步返回结果 + 当前head的值
```python
from typing import List
# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        return self.reversePrint(head.next) + [head.val] if head else []
```

***
剑指 Offer 07. 重建二叉树

    * 构建字典存储节点与对应中序idx
    * 根据前序中序的左右索引构建二叉树
    * 如果前序的左索引大于右索引，则返回None
    * 前序根节点索引为前序左（前序：根左右）
    * 使用前序中根节点索引，在idx字典中获取中序根节点索引
    * 构建根节点
    * 根据中序列表中根节点索引计算左侧子树长度
    * 调整前序中序的左右指针，构建左右子树
```python
from typing import List
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        def Tree(preorder_l, preorder_r, inorder_l, inorder_r):
            if preorder_l > preorder_r:
                return None

            preorder_root = preorder_l
            inorder_root = index[preorder[preorder_root]]

            root = TreeNode(preorder[preorder_root])
            subtree_span = inorder_root - inorder_l

            root.left = Tree(preorder_l + 1, preorder_r + subtree_span, inorder_l, inorder_root - 1)
            root.right = Tree(preorder_l + subtree_span + 1, preorder_r, inorder_root + 1, inorder_r)

            return root

        n = len(preorder)
        index = {element: i for i, element in enumerate(inorder)}
        return Tree(0, n - 1, 0, n - 1)
```

***
剑指 Offer 09. 用两个栈实现队列

    * 构建栈AB
    * 将A作为队尾栈，即将元素入栈即可
    * 将B作为队首栈，如果B不为空，则B出栈作为队首，如果为空且A也为空，则返回-1，如果B为空A不为空，则A倒叙入栈B，返回B的栈顶
```python
class CQueue:
    def __init__(self):
        self.A, self.B = [], []

    def appendTail(self, value: int) -> None:
        self.A.append(value)

    def deleteHead(self) -> int:
        if self.B:
            return self.B.pop()
        elif not self.A:
            return -1
        else:
            while self.A:
                self.B.append(self.A.pop())
            return self.B.pop()
```

***
剑指 Offer 10- I. 斐波那契数列

    * 初始化a=0, b=1
    * 遍历n，每次根据规则更新ab
    * 返回a % 1000000007
```python
class Solution:
    def fib(self, n: int) -> int:
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a % 1000000007
```

***
剑指 Offer 10- II. 青蛙跳台阶问题

    * 初始化a=b=1
    * 遍历n次，每次更新ab，到达当前台阶方法数 = 到达上一阶方法数（再跳一阶） + 到达上两阶方法数（再跳两阶）
```python
class Solution:
    def numWays(self, n: int) -> int:
        a, b = 1, 1
        for _ in range(n):
            a, b = b, a + b
        return a % 1000000007
```

***
剑指 Offer 11. 旋转数组的最小数字

    * 二分查找，每次比较中值与右侧数字
    * 如果小于右侧，则说明最小值不在中值右区间，改变右指针（注意防止恰好中值为最小值被忽略）
    * 如果大于右侧，则说明最小值不在中值左区间，改变左指针
    * 如果等于右侧，则不能直接定位最小值，因为会有重复，需要向左移动右指针
```python
from typing import List
class Solution:
    def minArray(self, numbers: List[int]) -> int:
        left, right = 0, len(numbers) - 1
        while left < right:
            mid = left + (right - left) // 2
            if numbers[mid] < numbers[right]:
                right = mid
            elif numbers[mid] > numbers[right]:
                left = mid + 1
            else:
                right -= 1
        return numbers[left]
```

***
剑指 Offer 12. 矩阵中的路径

    * 构建搜索方向数组
    * visited存放以当前起始位置，所搜索过的位置
    * 遍历每一个位置，是否满足条件
    * 如果当前位置不等于单词中对应字母，则F
    * 如果k达到单词长度，则T
    * 将当前位置填入visited
    * 搜索当前位置的四个方向，如果新位置满足条件，且不在visited中
    * 判断当前位置与单词的后续字母
    * 将该遍历位置移出visited，开始新位置的遍历
```python
from typing import List
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        def check(i, j, k):
            if board[i][j] != word[k]:
                return False
            if k == len(word) - 1:
                return True

            visited.add((i, j))
            result = False
            for di, dj in directions:
                new_i, new_j = i + di, j + dj
                if 0 <= new_i < r and 0 <= new_j < c:
                    if (new_i, new_j) not in visited:
                        if check(new_i, new_j, k + 1):
                            result = True
                            break
            visited.remove((i, j))
            return result

        r, c = len(board), len(board[0])
        visited = set()
        for i in range(r):
            for j in range(c):
                if check(i, j, 0):
                    return True
        return False
```

***
剑指 Offer 13. 机器人的运动范围

    * 构建visited集合用于存放可以到达的位置，初始位置(0, 0)可到达
    * 遍历矩阵，如果左或上可到达并且当前位置可到达，则当前位置可到达，加入visited（保持搜索范围连通性）
    * 定义digitsum计算当前坐标的数字和
```python
class Solution:
    def movingCount(self, m: int, n: int, k: int) -> int:
        def digitsum(n):
            ans = 0
            while n:
                ans += n % 10
                n //= 10
            return ans

        visited = set()
        visited.add((0, 0))
        for i in range(m):
            for j in range(n):
                if ((i - 1, j) in visited or (i, j - 1) in visited) and digitsum(i) + digitsum(j) <= k:
                    visited.add((i, j))
        return len(visited)
```

***
剑指 Offer 14- I. 剪绳子

    * 构建dp表，存放不同长度的最大结果
    * 初始化长度为2时结果为1
    * 遍历长度，对每个长度遍历第一次切分点
    * 更新dp表，如果选择不进行第二次切分，则结果为j * (i - j)
    * 如果进行第二次切分，则为j * dp[i - j]
```python
class Solution:
    def cuttingRope(self, n: int) -> int:
        dp = [0] * (n + 1)
        dp[2] = 1
        for i in range(3, n + 1):
            for j in range(2, i):
                dp[i] = max(dp[i], max(j * (i - j), j * dp[i - j]))
        return dp[n]
```

***
剑指 Offer 14- II. 剪绳子 II

    * 构建dp表，存放不同长度的最大结果
    * 初始化长度为2时结果为1
    * 遍历长度，对每个长度遍历第一次切分点
    * 更新dp表，如果选择不进行第二次切分，则结果为j * (i - j)
    * 如果进行第二次切分，则为j * dp[i - j]
```python
class Solution:
    def cuttingRope(self, n: int) -> int:
        if n <= 3:
            return n - 1
        dp = [0] * (n + 1)

        dp[2] = 1
        for i in range(3, n + 1):
            for j in range(2, i):
                dp[i] = max(dp[i], max(j * (i - j), j * dp[i - j]))
        return dp[n] % 1000000007
```

***
剑指 Offer 15. 二进制中1的个数

    * res存放结果
    * 循环，每次使用n&(n−1)消除最右边1，更新res
    * 直到n=0停止
```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        res = 0
        while n:
            res += 1
            n &= (n - 1)
        return res
```

***
剑指 Offer 16. 数值的整数次方

    * res存放结果，如果n<0，则将x取倒数即可将n转为正数
    * 循环，每次判断n能否整除2，及x能否转为x的平方
    * 如果可以，则记录当前多出的x
    * 更新x为x**2
    * 更新n为n//2
```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if x == 0:
            return 0
        res = 1
        if n < 0:
            x, n = 1 / x, -n
        while n:
            if n % 2 == 1:
                res *= x
            x *= x
            n //= 2
        return res
```

***
剑指 Offer 17. 打印从1到最大的n位数

    1. 初始化nine统计目前出现的9，以及start指定数字开始索引（排除递归时前面无用的0）
    2. 初始化num为所有的位数，res记录结果
    3. 递归num中的位数
    4. 如果位数x=n，说明满足条件，从start开始取数字，放入res（忽略0）
    5. 如果恰好到数字全是9，则start想左移动一位（说明数字进位，可以少考虑一个左侧0）
    6. 遍历0到10，如果恰好为9，则更新nine
    7. 更新num中对应位置数字为当前数字，进行递归
    8. 遍历结束后还原上一位统计过的9
```python
from typing import List
class Solution:
    def printNumbers(self, n: int) -> List[int]:
        self.nine = 0
        self.start = n - 1
        def dfs(x):
            if x == n:
                s = ''.join(num[self.start:])
                if s != '0':
                    res.append(int(s))
                if n - self.start == self.nine:
                    self.start -= 1
                return
            for i in range(10):
                if i == 9:
                    self.nine += 1
                num[x] = str(i)
                dfs(x + 1)
            self.nine -= 1

        num = [0] * n
        res = []
        dfs(0)
        return res
```

***
剑指 Offer 18. 删除链表的节点

    * 如果头节点即为目标节点，直接返回下个节点
    * 初始pre和cur
    * 遍历，如果cur存在且cur的值不为目标值，则更新pre和cur
    * 如果找到目标值，则删除目标节点
```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution:
    def deleteNode(self, head: ListNode, val: int) -> ListNode:
        if head.val == val:
            return head.next

        pre, cur = head, head.next
        while cur and cur.val != val:
            pre, cur = cur, cur.next

        if cur:
            pre.next = cur.next

        return head
```

***
剑指 Offer 19. 正则表达式匹配

    * 构建dp表，存放s中前i个字符是否与p中前j个匹配
    * 初始化dp[0][0]=T
    * 遍历，其中由于p为空时，s任何值都为F，所有p从1开始遍历
    * 如果p中元素为*，则判断上一个字符与s中当前字符是否匹配（其中.一定为T，s为空时一定不匹配）
    * 如果匹配，则返回s中去掉该字符的匹配结果与p中去掉该正则组合的匹配结果的|
    * 如果不匹配，则返回p中去掉该正则组合的匹配结果
    * 如果p中元素不为*，即为字符或.，则判断s与p中当前字符是否匹配
    * 如果匹配，则为dp[i - 1][j - 1]的匹配结果
    * 如果不匹配，则为F
    * 匹配中，如果没遇到*，那p有字符，s为空时一定不匹配
    * 如果遇到*，那上一个字符与空串s匹配直接返回F，跳转到比p中去掉这个组合的结果（如果为空串则为T）
```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)

        def matches(i, j):
            if i == 0:
                return False
            if p[j - 1] == '.':
                return True
            return s[i - 1] == p[j - 1]

        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        for i in range(m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    if matches(i, j - 1):
                        dp[i][j] = dp[i - 1][j] | dp[i][j - 2]
                    else:
                        dp[i][j] = dp[i][j - 2]
                else:
                    if matches(i, j):
                        dp[i][j] = dp[i - 1][j - 1]
        return dp[m][n]
```

***
剑指 Offer 20. 表示数值的字符串

    * 定义9种状态：
        0. 开始的空格
        1. 幂符号前的正负号
        2. 小数点前的数字
        3. 小数点、小数点后的数字
        4. 当小数点前为空格时，小数点、小数点后的数字
        5. 幂符号
        6. 幂符号后的正负号
        7. 幂符号后的数字
        8. 结尾的空格
    * 每种状态设置可行的转移状态
    * 设置四种情况（正负号-s、数字-d、e/E-e、空格/.-保持不变）
    * 初始状态为0
    * 遍历字符串，进行状态转移
    * 如果当前状态不在上个状态的可行下个状态种，则F
    * 更新当前状态
    * 如果最终状态在2378中，则T
```python
class Solution:
    def isNumber(self, s: str) -> bool:
        states = [
            {' ': 0, 's': 1, 'd': 2, '.': 4},
            {'d': 2, '.': 4},
            {'d': 2, '.': 3, 'e': 5, ' ': 8},
            {'d': 3, 'e': 5, ' ': 8},
            {'d': 3},
            {'s': 6, 'd': 7},
            {'d': 7},
            {'d': 7, ' ': 8},
            {' ': 8}
        ]

        p = 0
        for c in s:
            if '0' <= c <= '9':
                t = 'd'
            elif c in '+-':
                t = 's'
            elif c in 'eE':
                t = 'e'
            elif c in ' .':
                t = c
            else:
                t = '?'

            if t not in states[p]:
                return False

            p = states[p][t]
        return p in (2, 3, 7, 8)
```

***
剑指 Offer 21. 调整数组顺序使奇数位于偶数前面

    * 定义左右指针
    * 左指针从左向右寻找偶数，右指针从右向左寻找奇数
    * 交换左右指针
```python
from typing import List
class Solution:
    def exchange(self, nums: List[int]) -> List[int]:
        left, right = 0, len(nums) - 1
        while left < right:
            while left < right and nums[left] % 2 == 1:
                left += 1
            while left < right and nums[right] % 2 == 0:
                right -= 1
            nums[left], nums[right] = nums[right], nums[left]
        return nums
```

***
剑指 Offer 22. 链表中倒数第k个节点

    * 定义快慢指针
    * 快指针先走k步
    * 快慢指针共同前进，到快指针指向结尾，返回慢指针
```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
        fast, slow = head, head
        for _ in range(k):
            fast = fast.next

        while fast:
            fast = fast.next
            slow = slow.next
        return slow
```

***
剑指 Offer 24. 反转链表

    * 如果next为None，则返回
    * 递归（递归到倒数第二个节点）
    * head.next.next = head（下一个节点的下一个节点为当前节点，相当于翻转）
    * 下一个节点为None
```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if head == None or head.next == None:
            return head

        cur = self.reverseList(head.next)
        head.next.next = head
        head.next = None
        return cur
```

***
剑指 Offer 25. 合并两个排序的链表

    * 如果l1或者l2任何一个为None，返回另一个（递归终止条件）
    * 判断l1与l2的val大小，返回较小的一个（因为是上一个节点连接过来）
    * 将next与另一个节点再进行merge（期待返回里面较小的一个）
```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1:
            return l2
        if not l2:
            return l1

        if l1.val <= l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2
```

***
剑指 Offer 26. 树的子结构

    * 如果任何一棵树为空，则返回F
    * 前序遍历A树，每次与B的根节点做匹配（根节点、左子节点、右子节点，取或）
    * 判断A树当前节点与B的根节点对应结构是否相同
    * 如果B为空，则B与A对应子结构相同，为T
    * 如果A为空，或当前节点值与B的当前节点值不相等，则为F
    * 返回A.left, B.left与A.right, B.right
```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
        def recur(A, B):
            if not B:
                return True
            if not A or A.val != B.val:
                return False
            return recur(A.left, B.left) and recur(A.right, B.right)

        if not A or not B:
            return False

        return recur(A, B) or self.isSubStructure(A.left, B) or self.isSubStructure(A.right, B)
```

***
剑指 Offer 27. 二叉树的镜像

    * 递归至叶子节点
    * 翻转叶子节点
```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def mirrorTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return root

        left = self.mirrorTree(root.left)
        right = self.mirrorTree(root.right)
        root.left, root.right = right, left
        return root
```

***
剑指 Offer 28. 对称的二叉树

    * 如果没有根节点，则直接返回T
    * 递归左右子节点
    * 如果左右子节点都没有，则为T
    * 如果一个有一个没有，则为F
    * 如果都有，则比较值
    * 返回左左比右右，左右比右左
```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True

        def dfs(left, right):
            if not left and not right:
                return True
            elif not left or not right:
                return False
            elif left.val != right.val:
                return False
            return dfs(left.left, right.right) and dfs(left.right, right.left)
        return dfs(root.left, root.right)
```

***
剑指 Offer 29. 顺时针打印矩阵

    * 按逆时针顺序逐层遍历
    * 每次移动边指针，并判断是否结束
```python
from typing import List
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix:
            return []
        l, r, t, b = 0, len(matrix[0]) - 1, 0, len(matrix) - 1
        res = []

        while True:
            for i in range(l, r + 1):
                res.append(matrix[t][i])
            t += 1
            if t > b:
                break

            for i in range(t, b + 1):
                res.append(matrix[i][r])
            r -= 1
            if l > r:
                break

            for i in range(r, l - 1, -1):
                res.append(matrix[b][i])
            b -= 1
            if t > b:
                break

            for i in range(b, t - 1, -1):
                res.append(matrix[i][l])
            l += 1
            if l > r:
                break
        return res
```

***
剑指 Offer 30. 包含min函数的栈

```python
class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.A, self.B = [], []

    def push(self, x: int) -> None:
        self.A.append(x)
        if not self.B or self.B[-1] >= x:
            self.B.append(x)

    def pop(self) -> None:
        element = self.A.pop()
        if element == self.B[-1]:
            self.B.pop()

    def top(self) -> int:
        return self.A[-1]

    def min(self) -> int:
        return self.B[-1]
```

***
剑指 Offer 31. 栈的压入、弹出序列
    * 构建辅助栈，模拟压入弹出操作，结束后如果栈为空则为T
    * 遍历压入列表，数字入栈
    * 如果栈不为空，循环比较栈顶元素与出栈序列的元素列表
    * 如果相等，则表示当前操作应为出栈，将栈顶元素出栈
```python
from typing import List
class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        stack = []
        i = 0
        for num in pushed:
            stack.append(num)
            while stack and stack[-1] == popped[i]:
                stack.pop()
                i += 1
        return not stack
```

***
剑指 Offer 32 - I. 从上到下打印二叉树
    * res记录结果
    * 构建队列，初始化为root
    * 如果队列有元素，进行循环
    * 队首出队，加入res，判断如果当前元素有左节点或右节点，则入队
```python
from typing import List
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[int]:
        if not root:
            return []

        res = []
        que = [root]
        while que:
            first = que.pop(0)
            res.append(first.val)
            if first.left:
                que.append(first.left)
            if first.right:
                que.append(first.right)
        return res
```

***
剑指 Offer 32 - II. 从上到下打印二叉树 II
    * 如果没有根节点，返回[]
    * res储存结果
    * 构建队列que，root入队
    * 遍历队列（队列不断更新）
    * 遍历队列长度（tmp存放该层所有节点值）
    * 元素出队，并存放其值（遍历结束后，该层所有节点出队）
    * 将其左右子节点入队，作为下一层的遍历对象
```python
from typing import List
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []

        res = []
        que = [root]
        while que:
            size = len(que)
            tmp = []
            for _ in range(size):
                first = que.pop(0)
                tmp.append(first.val)
                if first.left:
                    que.append(first.left)
                if first.right:
                    que.append(first.right)
            res.append(tmp)
        return res
```

***
剑指 Offer 32 - III. 从上到下打印二叉树 III
    * 如果没有根节点，返回[]
    * res储存结果
    * 构建队列que，root入队
    * 遍历队列（队列不断更新）
    * 遍历队列长度（tmp存放该层所有节点值）
    * 元素出队，并存放其值（遍历结束后，该层所有节点出队）
    * 将其左右子节点入队，作为下一层的遍历对象
    * 分奇偶层将tmp加入res
```python
from typing import List
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []

        res = []
        que = [root]
        while que:
            size = len(que)
            tmp = []
            for _ in range(size):
                first = que.pop(0)
                tmp.append(first.val)
                if first.left:
                    que.append(first.left)
                if first.right:
                    que.append(first.right)
            res.append(tmp[::-1] if len(res) % 2 else tmp)
        return res
```

***
剑指 Offer 33. 二叉搜索树的后序遍历序列
    * 设置左右判断区间，右区间代表根节点（后序遍历）
    * 指针p指向左区间，寻找第一个大于根节点的索引（保证左子树都小于根节点）
    * 取l到当前m=p为左子树，取m到r为右子树
    * 继续移动指针p，保证右子树都大于根节点
    * 如果最终指针不指向r（说明右子树不都大于根节点），则为F，并递归判断左右子树
```python
from typing import List
class Solution:
    def verifyPostorder(self, postorder: List[int]) -> bool:
        def dfs(l, r):
            if l >= r:
                return True

            p = l
            while postorder[p] < postorder[r]:
                p += 1
            m = p
            while postorder[p] > postorder[r]:
                p += 1
            return p == r and dfs(l, m - 1) and dfs(m, r)
        return dfs(0, len(postorder) - 1)
```

***
剑指 Offer 34. 二叉树中和为某一值的路径
    * res存放所有路径结果，path存放路径
    * 递归，如果到达叶子节点则返回
    * 将当前节点填入path，并将target-当前节点值，作为还需要的值
    * 如果到达叶子节点且路径值恰好等于target，则将path填入res
    * 递归左右节点，pop返回上一步状态
```python
from typing import List
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def pathSum(self, root: TreeNode, target: int) -> List[List[int]]:
        res = []
        path = []

        def dfs(root, target):
            if not root:
                return
            path.append(root.val)
            target -= root.val
            if not root.left and not root.right and target == 0:
                res.append(path[:])
            dfs(root.left, target)
            dfs(root.right, target)
            path.pop()
        dfs(root, target)
        return res
```

***
剑指 Offer 35. 复杂链表的复制
    * 构建hash表存放节点对应以自身值构建的无连接新节点的键值对
    * 遍历链表，更新哈希表
    * 遍历链表，每次构建next与random
```python
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random

class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head:
            return
        hash_table = dict()
        cur = head

        while cur:
            hash_table[cur] = Node(cur.val)
            cur = cur.next

        cur = head
        while cur:
            hash_table[cur].next = hash_table.get(cur.next)
            hash_table[cur].random = hash_table.get(cur.random)
            cur = cur.next
        return hash_table[head]
```

***
剑指 Offer 36. 二叉搜索树与双向链表
    * 如果没有root返回
    * 设置前驱节点pre
    * 中序遍历，每次操作如果pre为空则记录cur为head，否则pre右指向cur，cur左指向pre
    * 更新pre为cur
    * head与pre首尾连接
    * 返回head
```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
"""
class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        def dfs(cur):
            if not cur:
                return
            dfs(cur.left)
            if self.pre:
                self.pre.right = cur
                cur.left = self.pre
            else:
                self.head = cur
            self.pre = cur
            dfs(cur.right)

        if not root:
            return
        self.pre = None
        dfs(root)
        self.head.left = self.pre
        self.pre.right = self.head
        return self.head
```

***
剑指 Offer 37. 序列化二叉树
    * 序列化时，遇到空节点，返回null
    * 递归，根节点 + , + 左子节点 + , + 右子节点，完成序列化（获取类似前序遍历结果）
    * 反序列化时，先将字符串转成list
    * 递归，每次出队节点为root，在按顺序还原左右节点，返回root
```python
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return None
        return str(root.val) + ',' + str(self.serialize(root.left)) + ',' + str(self.serialize(root.right))


    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """

        def dfs(datalist):
            val = datalist.pop(0)
            if val == 'None':
                return None

            root = TreeNode(int(val))
            root.left = dfs(datalist)
            root.right = dfs(datalist)
            return root

        datalist = data.split(',')
        return dfs(datalist)
```

***
剑指 Offer 38. 字符串的排列
    * 将字符串转为list
    * 维护需要遍历的子数组第一个位置索引first
    * 如果为n-1，则填入ans
    * 构建dic集合，存放已经出现的元素
    * 遍历first到n，每次与first交换、回溯first+1、复原（相当于遍历所有组合方式）
    * 其中要对已经出现过的元素进行剪枝，新元素加到集合中
```python
from typing import List
class Solution:
    def permutation(self, s: str) -> List[str]:
        def backtrack(first):
            if first == n - 1:
                ans.append(''.join(s_li))
                return
            dic = set()
            for i in range(first, n):
                if s_li[i] in dic:
                    continue
                dic.add(s_li[i])
                s_li[first], s_li[i] = s_li[i], s_li[first]
                backtrack(first + 1)
                s_li[i], s_li[first] = s_li[first], s_li[i]

        n = len(s)
        s_li = list(s)
        ans = []
        backtrack(0)
        return ans
```

***
剑指 Offer 39. 数组中出现次数超过一半的数字
    * 遍历数组，每次假设当前数字为众数
    * 维护votes记录众数得分，如果后续数与众数相同+1，不同-1
    * 如果votes变为0，忽略前面数字，再假设下一个数字为众数
    * 如果假设数字为真实众数，与非众数消掉后，不影响后续众数判断
    * 如果假设数字不为真实众数，与其他任何数字消掉后，也不影响后续众数判断
```python
from typing import List
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        votes = 0
        for num in nums:
            if votes == 0:
                x = num

            if num == x:
                votes += 1
            else:
                votes -= 1
        return x
```

***
剑指 Offer 40. 最小的k个数
    1. 如果k大于等于数组长度，则返回数组
    2. 进行快速排序获取索引
    3. 比较该索引与k大小，进行调整
```python
from typing import List
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        if k >= len(arr):
            return arr
        left = 0
        right = len(arr) - 1
        while True:
            index = self.__partition(arr, left, right)
            if index == k:
                return arr[: index]
            elif index < k:
                left = index + 1
            else:
                right = index - 1

    def __partition(self, nums, left, right):
        import random
        random_index = random.randint(left, right)
        nums[random_index], nums[left] = nums[left], nums[random_index]

        pivot = nums[left]
        while left < right:
            while left < right and nums[right] >= pivot:
                right -= 1
            nums[left] = nums[right]
            while left < right and nums[left] <= pivot:
                left += 1
            nums[right] = nums[left]
        nums[left] = pivot
        return left
```

***
剑指 Offer 41. 数据流中的中位数
    * 构建小顶堆A，用于存放较大的一半元素，大顶堆B用于存放较小的一半元素（方便根据两个堆顶直接取中位数）
    * 如果当前两个堆中元素个数相同，此时要向A中添加新元素，先将元素放入B中，再将B的堆顶弹出放入A中（保证A中数不小于B）
    * 如果当前两个堆中元素个数不同，此时要向B中添加新元素，先将元素放入A中，再将A的堆顶弹出放入B中（保证B中数不大于A）
    * 如果两个堆中元素个数相同，取两堆顶平均为中位数，否则取A的堆顶（因为始终保持A中数字个数不小于B）
```python
from heapq import *
class MedianFinder:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.A = []
        self.B = []


    def addNum(self, num: int) -> None:
        if len(self.A) != len(self.B):
            heappush(self.A, num)
            heappush(self.B, -heappop(self.A))
        else:
            heappush(self.B, -num)
            heappush(self.A, -heappop(self.B))

    def findMedian(self) -> float:
        return self.A[0] if len(self.A) != len(self.B) else (self.A[0] - self.B[0]) / 2.0
```

***
剑指 Offer 42. 连续子数组的最大和
    * 动态规划，遍历数组，每个数字表示以当前数字为结尾的连续数组最大和
    * 每次更新当前数字，使用前面数字的和
    * 如果前面数字和为负数，则直接取当前数字，否则加上前面数字和
    * 取dp表中最大的结果返回
```python
from typing import List
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        for i in range(1, len(nums)):
            nums[i] += max(nums[i - 1], 0)
        return max(nums)
```

***
剑指 Offer 43. 1～n 整数中 1 出现的次数
    * 将数字分为当前位数字和高低位数字
    * 如果当前数字为0，则此位出现1的次数由高位决定，high * digit
    * 如果当前数字为1，则此为出现1的次数由高低位共同决定，high * digit + low + 1
    * 如果当前数字为2到9，则此位出现1的次数由高位决定，(high + 1) * digit
    * 更新当前、高低位数字，以及位数
```python
class Solution:
    def countDigitOne(self, n: int) -> int:
        res = 0
        digit = 1
        high = n // 10
        cur = n % 10
        low = 0

        while high != 0 or cur != 0:
            if cur == 0:
                res += high * digit
            elif cur == 1:
                res += high * digit + low + 1
            else:
                res += (high + 1) * digit

            low += cur * digit
            cur = high % 10
            high //= 10
            digit *= 10
        return res
```

***
剑指 Offer 44. 数字序列中某一位的数字
    * 初始话当前数位，当前数位开始数字，当前数位总数位数
    * 找到当前数位，使用n不断减去低数位的总数位数，直到小于等于0，每次更新digit、start、count
    * 确定当前数字，找到数位后，从当前start开始，加上剩余数位数-1 // 当前数位每个数字对应数位数
    * 确定当前数字中的位数，剩余数位数对当前数位每个数字对应数位数取余
    * 返回结果，先将num转为str，取索引后再转为int
```python
class Solution:
    def findNthDigit(self, n: int) -> int:
        digit, start, count = 1, 1, 9
        while n > count:
            n -= count
            start *= 10
            digit += 1
            count = 9 * start * digit
        num = start + (n - 1) // digit
        return int(str(num)[(n - 1) % digit])
```

***
剑指 Offer 45. 把数组排成最小的数
    * 定义新的判断大小规则，x+y > y+x，说明x大于y
    * 快速排序，根据新规则进行排序
```python
from typing import List
class Solution:
    def minNumber(self, nums: List[int]) -> str:
        def partition(li, l, r):
            tmp = li[l]
            while l < r:
                while l < r and tmp + strs[r] <= strs[r] + tmp:
                    r -= 1
                strs[l] = strs[r]
                while l < r and strs[l] + tmp <= tmp + strs[l]:
                    l += 1
                strs[r] = strs[l]
            li[l] = tmp
            return l

        def quick_sort(li, l, r):
            if l < r:
                mid = partition(li, l, r)
                quick_sort(li, l, mid - 1)
                quick_sort(li, mid + 1, r)

        strs = [str(num) for num in nums]
        quick_sort(strs, 0, len(strs) - 1)
        return "".join(strs)
```

***
剑指 Offer 46. 把数字翻译成字符串
    * 构建dp，使用ab存放前两个位置情况
    * 上一个数和当前数组合在10-25区间即为合理，0-10和25-99不合理
    * 如果上一个数和当前数组合可以被翻译，则当前数情况可由上一个数之前情况（加组合）+上一个数情况（加当前数）来表示
    * 如果组合不合法，则只能由上一个数情况（加当前数）来表示
    * 遍历数字长度，更新状态
```python
class Solution:
    def translateNum(self, num: int) -> int:
        s = str(num)
        a = b = 1
        for i in range(2, len(s) + 1):
            a, b = b, (a + b if '10' <= s[i - 2: i] <= '25' else b)
        return b
```

***
剑指 Offer 47. 礼物的最大价值
    * 将给定表格转为dp表，每个元素为左侧元素和上侧元素中较大的一个+当前元素
    * 初始化第一行第一列，遍历更新整个矩阵
```python
from typing import List
class Solution:
    def maxValue(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        for j in range(1, n):
            grid[0][j] += grid[0][j - 1]
        for i in range(1, m):
            grid[i][0] += grid[i - 1][0]

        for i in range(1, m):
            for j in range(1, n):
                grid[i][j] += max(grid[i][j - 1], grid[i - 1][j])
        return grid[-1][-1]
```

***
剑指 Offer 48. 最长不含重复字符的子字符串
    * 构建指针rk，ans记录结果长度，occ为当前维护的不重复字符集合
    * 遍历字符串，除了首位以外，每次从occ中删除字符串中的上一个字符（因为已经计算完以该字符起始的最大长度了）
    * 持续移动指针rk，如果s[rk]不在occ中，将其加入进去（rk为以当前i起始的最大连续不重复字符串的结尾）
    * 使用rk-i更新ans
```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        ans = 0
        rk = 0
        occ = set()
        n = len(s)
        for i in range(n):
            if i != 0:
                occ.remove(s[i - 1])
            while rk < n and s[rk] not in occ:
                occ.add(s[rk])
                rk += 1
            ans = max(ans, rk - i)
        return ans
```

***
剑指 Offer 49. 丑数
    * 构建dp表，存放第i个丑数
    * abc三指针，代表235
    * 遍历，每次计算三指针各自乘对应数
    * 更新dp为其中较小的一个，对应指针更新
```python
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        dp = [1] * n
        a, b, c = 0, 0, 0
        for i in range(1, n):
            n2, n3, n5 = dp[a] * 2, dp[b] * 3, dp[c] * 5
            dp[i] = min(n2, n3, n5)
            if dp[i] == n2:
                a += 1
            elif dp[i] == n3:
                b += 1
            elif dp[i] == n5:
                c += 1
        return dp[-1]
```

***
剑指 Offer 50. 第一个只出现一次的字符
    * 构建hash表，存放每个字符及其出现情况
    * 遍历字符串，如果hash表中没有，则填入T，如果已有，则设为F
    * 遍历哈希表，如果为T，则弹出
```python
class Solution:
    def firstUniqChar(self, s: str) -> str:
        hash_table = dict()
        for c in s:
            if c in hash_table:
                hash_table[c] = False
            else:
                hash_table[c] = True

        for k, v in hash_table.items():
            if v:
                return k
        return ' '
```

***
剑指 Offer 51. 数组中的逆序对
    * 构建tmp暂存当前需要分裂的左右数组
    * 归并排序，每次归并用时进行判断
    * 如果左数组到达边界，填入右元素，更新右指针
    * 如果右数组到达边界，填入左元素，更新左指针
    * 如果当前左元素小于等于当前右元素，说明没有逆序对，更新左指针
    * 如果当前左元素大于当前右元素，存在逆序对，更新右指针，res（因为左右数组各自有序，即统计当前元素后面元素即可）
```python
from typing import List
class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        def merge_sort(l, r):
            if l >= r:
                return 0
            m = l + (r - l) // 2
            res = merge_sort(l, m) + merge_sort(m + 1, r)
            i, j = l, m + 1
            tmp[l: r + 1] = nums[l: r + 1]
            for k in range(l, r + 1):
                if i == m + 1:
                    nums[k] = tmp[j]
                    j += 1
                elif j == r + 1 or tmp[i] <= tmp[j]:
                    nums[k] = tmp[i]
                    i += 1
                else:
                    nums[k] = tmp[j]
                    j += 1
                    res += m + 1 - i
            return res

        tmp = [0] * len(nums)
        return merge_sort(0, len(nums) - 1)
```

***
剑指 Offer 52. 两个链表的第一个公共节点
    * 定义双指针，任何一个先到达终点，立刻指向另一个指针起点
    * 两指针相交时，恰好为相交点
```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        A, B = headA, headB
        while A != B:
            A = A.next if A else headB
            B = B.next if B else headA
        return A
```

***
剑指 Offer 53 - I. 在排序数组中查找数字 I
    * 二分查找，先找区间右侧，即当mid为target则移动左
    * 判断右侧数字是否为target，如果不是直接返回
    * 重置l，保留r
    * 找区间左侧，即当mid为target则移动右
```python
from typing import List
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = l + (r - l) // 2
            if nums[mid] <= target:
                l = mid + 1
            else:
                r = mid - 1
        span_r = l

        if r >= 0 and nums[r] != target:
            return 0

        l = 0
        while l <= r:
            mid = l + (r - l) // 2
            if nums[mid] >= target:
                r = mid - 1
            else:
                l = mid + 1
        span_l = r
        return span_r - span_l - 1
```

***
剑指 Offer 53 - II. 0～n-1中缺失的数字
    * 二分查找，当前数值不等于索引的元素
```python
from typing import List
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = l + (r - l) // 2
            if nums[mid] == mid:
                l = mid + 1
            else:
                r = mid - 1
        return l
```

***
剑指 Offer 54. 二叉搜索树的第k大节点
    * 倒序中序遍历
    * 如果k降为0，直接return
    * 每次k-1，为0时记录结果
```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def kthLargest(self, root: TreeNode, k: int) -> int:
        def dfs(root):
            if not root:
                return
            dfs(root.right)

            if self.k == 0:
                return
            self.k -= 1
            if self.k == 0:
                self.res = root.val

            dfs(root.left)
        self.k = k
        self.res = 0
        dfs(root)
        return self.res
```

***
剑指 Offer 55 - I. 二叉树的深度
    * 如果没有根节点，返回0（递归终止条件）
    * 递归左侧最大深度，右侧最大深度
    * 两侧最大深度中的max + 1（根节点）
```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        else:
            left_height = self.maxDepth(root.left)
            right_height = self.maxDepth(root.right)
            return max(left_height, right_height) + 1
```

***
剑指 Offer 55 - II. 平衡二叉树
    * 递归，自下而上判断
    * 每次更新左右节点深度
    * 比较左右节点深度，如果大于1则返回-1，提前中值
```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        def dfs(root):
            if not root:
                return 0
            left = dfs(root.left)
            if left == -1:
                return -1
            right = dfs(root.right)
            if right == -1:
                return -1
            return max(left, right) + 1 if abs(left - right) <= 1 else -1
        return dfs(root) != -1
```

***
剑指 Offer 56 - I. 数组中数字出现的次数
    * x,y用于保存结果
    * n记录两个数字的异或结果，m记录两个数字首位不同的二进制位
    * 遍历数组进行异或，获取n
    * 迭代m逐渐向左移动，寻找首位不同位
    * 遍历数组，根据不同位进行分组，计算异或结果
```python
from typing import List
class Solution:
    def singleNumbers(self, nums: List[int]) -> List[int]:
        x, y = 0, 0
        n = 0
        m = 1
        for num in nums:
            n ^= num

        while n & m == 0:
            m <<= 1

        for num in nums:
            if num & m:
                x ^= num
            else:
                y ^= num
        return [x, y]
```

***
剑指 Offer 56 - II. 数组中数字出现的次数 II
    * 构建counts存放32位每位的总1个数
    * 遍历数组，遍历32位数，更新counts
    * 反向遍历counts，对3取余数，返回十进制
```python
from typing import List
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        counts = [0] * 32
        for num in nums:
            for j in range(32):
                counts[j] += num & 1
                num >>= 1
        res = 0
        for i in range(32):
            res <<= 1
            res |= counts[31 - i] % 3
        return res if counts[31] % 3 == 0 else ~(res ^ 0xffffffff)
```

***
剑指 Offer 57. 和为s的两个数字
    * 双指针，遍历（排序数组）
    * 如果加和小于目标，则左指针移动，否则右指针移动
```python
from typing import List
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        i, j = 0, len(nums) - 1
        while i < j:
            s = nums[i] + nums[j]
            if s > target:
                j -= 1
            elif s < target:
                i += 1
            else:
                return [nums[i], nums[j]]
        return []
```

***
剑指 Offer 57 - II. 和为s的连续正数序列
    * 构建滑动窗口，记录窗口内数字和
    * 移动窗口，如果恰好位target，则加入res，移动左指针
    * 如果小于target，移动左指针
    * 如果大于target，移动右指针
```python
from typing import List
class Solution:
    def findContinuousSequence(self, target: int) -> List[List[int]]:
        i, j = 1, 2
        s = i + j
        res = []
        while i < j:
            if s == target:
                res.append(list(range(i, j + 1)))
            if s >= target:
                s -= i
                i += 1
            else:
                j += 1
                s += j
        return res
```

***
剑指 Offer 58 - I. 翻转单词顺序
    * 删除前后空格，并在尾部构建双指针
    * 遍历首指针位置，如果不为空格则移动，将指针区间加入res
    * 遇到空格则移动，忽略连续空格
    * 两指针对齐
```python
class Solution:
    def reverseWords(self, s: str) -> str:
        s = s.strip()
        i, j = len(s) - 1, len(s) - 1
        res = []
        while i >= 0:
            while i >= 0 and s[i] != ' ':
                i -= 1
            res.append(s[i + 1: j + 1])

            while s[i] == ' ':
                i -= 1
            j = i
        return ' '.join(res)
```

***
剑指 Offer 58 - II. 左旋转字符串
    * 切片拼接
```python
class Solution:
    def reverseLeftWords(self, s: str, n: int) -> str:
        return s[n:] + s[:n]
```

***
剑指 Offer 59 - I. 滑动窗口的最大值
    * 构建单调队列q（只需保证当前窗口最大的在队列中，且队列降序，因为左侧较小的之后不会用到）
    * 将第一个窗口逐个加入队列，保持队列降序（如果当前元素大于等于队尾，则将队尾逐个弹出）
    * 维护ans存放结果
    * 后续窗口逐个入队，保持队列降序
    * 每次窗口移动时，如果队首索引不在窗口内，则将队首弹出
    * 使用队首元素更新ans
```python
from typing import List
import collections
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if not nums or k == 0:
            return []
        n = len(nums)
        q = collections.deque()
        for i in range(k):
            while q and nums[i] >= nums[q[-1]]:
                q.pop()
            q.append(i)

        ans = [nums[q[0]]]
        for i in range(k, n):
            while q and nums[i] >= nums[q[-1]]:
                q.pop()
            q.append(i)

            while q[0] <= i - k:
                q.popleft()
            ans.append(nums[q[0]])
        return ans
```

***
剑指 Offer 59 - II. 队列的最大值
    * 维护正常队列和一个单调队列，单调队列保持递减
    * maxvalue返回单调队列首位元素
    * pushback的时候，单调队列保持递减
    * popfront时，判断单调队列队首如果相同则同步出队
```python
import queue
class MaxQueue:
    def __init__(self):
        self.queue = queue.Queue()
        self.deque = queue.deque()

    def max_value(self) -> int:
        return self.deque[0] if self.deque else -1

    def push_back(self, value: int) -> None:
        self.queue.put(value)
        while self.deque and self.deque[-1] < value:
            self.deque.pop()
        self.deque.append(value)

    def pop_front(self) -> int:
        if self.queue.empty():
            return -1
        val = self.queue.get()
        if val == self.deque[0]:
            self.deque.popleft()
        return val
```

***
剑指 Offer 60. n个骰子的点数
    * 构建dp表，表示当前层的各点数概率，初始化为1/6
    * 遍历各层，存放tmp为下一层个点数概率
    * 遍历当前层dp表，遍历首当前数字影响的后6各数字，更新tmp
    * 使用tmp更新dp
```python
from typing import List
class Solution:
    def dicesProbability(self, n: int) -> List[float]:
        dp = [1 / 6] * 6
        for i in range(2, n + 1):
            tmp = [0] * (5 * i + 1)
            for j in range(len(dp)):
                for k in range(6):
                    tmp[j + k] += dp[j] / 6
            dp = tmp
        return dp
```

***
剑指 Offer 61. 扑克牌中的顺子
    * 构建repeat集合判断是否出现重复（出现重复数字则无法构成顺子）
    * 记录最大最小值（当除了0外最大-最小<5则满足条件）
    * 遍历更新最大最小以及repeat集合
```python
from typing import List
class Solution:
    def isStraight(self, nums: List[int]) -> bool:
        repeat = set()
        num_max, num_min = 0, 14
        for num in nums:
            if num == 0:
                continue
            num_max = max(num_max, num)
            num_min = min(num_min, num)

            if num in repeat:
                return False
            repeat.add(num)
        return num_max - num_min < 5
```

***
剑指 Offer 62. 圆圈中最后剩下的数字
    * dp问题，f(n) = (f(n - 1) + t) % n，其中t=m%n，得f(n) = (f(n - 1) + m) % n
    * 遍历n，迭代更新
```python
class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        x = 0
        for i in range(2, n + 1):
            x = (x + m) % i
        return x
```

***
剑指 Offer 63. 股票的最大利润
    * 维护历史最低价，总最大利润
    * 遍历，每次更新历史最低价
    * 使用当前价格与历史最低价差更新最大利润
```python
from typing import List
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        minprice = float('inf')
        maxprofit = 0
        for price in prices:
            minprice = min(minprice, price)
            maxprofit = max(maxprofit, price - minprice)
        return maxprofit
```

***
剑指 Offer 64. 求1+2+…+n
    * 使用逻辑运算来控制递归的终止条件
    * A and B语句，如果A=F则不执行B
```python
class Solution:
    def __init__(self):
        self.res = 0
    def sumNums(self, n: int) -> int:
        n > 1 and self.sumNums(n - 1)
        self.res += n
        return self.res
```

***
剑指 Offer 65. 不用加减乘除做加法
    * 使用异或运算、与运算、位运算来计算
    * a + b = a ^ b + (a & b) << 1
    * 循环至b为0，即不在出现进位操作时，a即为结果
```python
class Solution:
    def add(self, a: int, b: int) -> int:
        x = 0xffffffff
        a, b = a & x, b & x
        while b != 0:
            a, b = (a ^ b), (a & b) << 1 & x
        return a if a <= 0x7fffffff else ~(a ^ x)
```

***
剑指 Offer 66. 构建乘积数组
    * 从左向右遍历数组，计算每个数字左侧的累计乘积
    * 从右向左遍历数组，计算每个数字右侧的累计乘积
    * 更新结果
```python
from typing import List
class Solution:
    def constructArr(self, a: List[int]) -> List[int]:
        n = len(a)
        b = [1] * n
        tmp = 1
        for i in range(1, n):
            b[i] = b[i - 1] * a[i - 1]

        for i in range(n - 2, -1, -1):
            tmp *= a[i + 1]
            b[i] *= tmp
        return b
```

***
剑指 Offer 67. 把字符串转换成整数
    * 初始化res、sign以及int_max, int_min, boundry
    * 如果str为空，直接返回0
    * 指针移动至首个非空格字符，如果全为空格则返回0
    * 判断首个非空格字符如果为-号则sign为-1，移动指针
    * 遍历字符串，如果当前字符不在0到9之间，直接跳出
    * 更新res = 10 * res + ord(c) - ord('0')
    * 判断res是否越界，进行特殊处理
    * 返回sign * res
```python
class Solution:
    def strToInt(self, str: str) -> int:
        res = 0
        i = 0
        sign = 1
        n = len(str)
        int_max, int_min, boundry = 2 ** 31 - 1, -2 ** 31, 2 ** 31 // 10
        if not str:
            return 0

        while str[i] == ' ':
            i += 1
            if i == n:
                return 0

        if str[i] == '-':
            sign = -1
        if str[i] in '+-':
            i += 1

        for c in str[i:]:
            if not '0' <= c <= '9':
                break
            if res > boundry or (res == boundry and c > '7'):
                return int_max if sign == 1 else int_min
            res = 10 * res + ord(c) - ord('0')
        return sign * res
```

***
剑指 Offer 68 - I. 二叉搜索树的最近公共祖先
    * 如果qp都在右子树，则判断右子树
    * 如果pq都在左子树，则判断左子树
    * 如果pq在一左一右，即为最近公共祖先，返回即可
```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        while root:
            if root.val < p.val and root.val < q.val:
                root = root.right
            elif root.val > p.val and root.val > q.val:
                root = root.left
            else:
                break
        return root
```

***
剑指 Offer 68 - II. 二叉树的最近公共祖先
    * 如果没有root（抵达叶子节点）或者root为p或q（找到p或q提取结束），则直接返回root
    * 递归左侧，递归右侧
    * 左右同时为空，则该子树不包含pq，返回
    * 均不为空，则该节点为最近公共祖先，直接返回
    * 一侧空一侧不空，此时该侧为p、q之一，或该侧即为最近公共祖先（下面包括另一节点）
```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root or root == p or root == q:
            return root

        l = self.lowestCommonAncestor(root.left, p, q)
        r = self.lowestCommonAncestor(root.right, p, q)
        if not l and not r:
            return
        if not l:
            return r
        if not r:
            return l
        return root
```