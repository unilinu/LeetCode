### [912\. 归并排序数组](https://leetcode-cn.com/problems/sort-an-array/)

Difficulty: **中等**


给你一个整数数组 `nums`，请你将该数组升序排列。

**示例 1：**

```
输入：nums = [5,2,3,1]
输出：[1,2,3,5]
```

**提示：**

1.  `1 <= nums.length <= 50000`
2.  `-50000 <= nums[i] <= 50000`


#### Solution

Language: ****

```cpp
#include<iostream>
#include<vector>

using namespace std;

class Solution {
public:
    vector<int> sortArray(vector<int> &nums) {
        sort(nums, 0, nums.size() - 1);
        return nums;
    }

private:
    void sort(vector<int> &nums, int low, int high) {
        if (low == high) return;
        int mid = (low + high) / 2;
        sort(nums, low, mid);
        sort(nums, mid + 1, high);
        merge(nums, low, mid, mid + 1, high);
    }

    void merge(vector<int> &nums, int ll, int lh, int rl, int rh) {
        // TODO: enhance performance
        vector<int> left(nums.begin() + ll, nums.begin() + lh + 1), right(nums.begin() + rl, nums.begin() + rh + 1);
        int l = 0, r = 0;
        while (ll <= rh) {  // WATCH: the skill here
            if(l==left.size()){
                nums[ll++]=right[r++];
                continue;
            }
            if(r==right.size()) {
                nums[ll++]=left[l++];
                continue;
            }
            nums[ll++] = left[l]<right[r] ? left[l++] : right[r++];
        }
    }
};

int main() {
    Solution s;
    vector<int> nums = {5, 2, 3, 1, 4, 3}, res(s.sortArray(nums));
    for (int i = 0; i < res.size(); ++i) {
        cout << res[i] << ' ';
    }

    return 0;
}
```

```go

```

