

# Code101

[toc]

## 1. 贪心

### 455 Assign cookies (easy)

```cpp
class Solution {
public:
    int findContentChildren(vector<int>& children, vector<int>& cookies) { 
        sort(children.begin(), children.end()); 
        sort(cookies.begin(), cookies.end()); 
        int child = 0, cookie = 0; 
        while (child < children.size() && cookie < cookies.size()) { 
            if (children[child] <= cookies[cookie]) 
                ++child; 
            ++cookie;
        }
        return child;
    }
};
```

### 135 Candy (hard)

- Solution 1

```go
// candy2 initializes every rating with 1 candy,
// and then from left to right to update by comparing with the left side
// and from right to left to update by comparing with the right side
func candy(ratings []int) int {
	candies := make([]int, len(ratings))
	for i := range candies {
		candies[i] = 1
	}

	for i := 1; i < len(ratings); i++ {
		if ratings[i] > ratings[i-1] && candies[i] <= candies[i-1] {
			candies[i] = candies[i-1] + 1
		}
	}
	for i := len(ratings) - 2; i >= 0; i-- {
		if ratings[i] > ratings[i+1] && candies[i] <= candies[i+1] {
			candies[i] = candies[i+1] + 1
		}
	}

	res := 0
	for _, i := range candies {
		res += i
	}

	return res
}
```



- Solution 2

```go
// candy2 initializes every rating with 1 candy,
// and then from the least to the most,
// make the ratings at left and right sides suited one by one
func candy2(ratings []int) int {
	greedy := make([]int, len(ratings))
	for i := range greedy {
		greedy[i] = i
	}
	sort.Slice(greedy, func(i, j int) bool {
		return ratings[greedy[i]] < ratings[greedy[j]]
	})

	candies := make([]int, len(ratings))
	for i := range candies {
		candies[i] = 1
	}
	for _, i := range greedy {
		// ensure left side
		if i > 0 && ratings[i-1] > ratings[i] && candies[i-1] <= candies[i] {
			candies[i-1] = candies[i] + 1
		}
		// ensure right side
		if i < len(ratings)-1 && ratings[i+1] > ratings[i] && candies[i+1] <= candies[i] {
			candies[i+1] = candies[i] + 1
		}
	}

	res := 0
	for _, val := range candies {
		res += val
	}

	return res
}
```



```cpp
class Solution {
public:
    int candy(vector<int>& ratings) {
        // ERROR
        int size=ratings.size();
        if(size<2) return size;
        //
        vector<int> candy(ratings.size(), 1);
        for(int i=1; i<ratings.size(); ++i){
            if(ratings[i]>ratings[i-1])
                candy[i]=candy[i-1]+1;
        }
        for(int i=ratings.size()-2; i>=0; --i){
            if(ratings[i]>ratings[i+1] && candy[i]<=candy[i+1])
                candy[i]=candy[i+1]+1;
        }
        // OR
        // return accumulate(num.begin(), num.end(), 0);
        for(int i=1;i<candy.size();++i)
            candy[0]+=candy[i];
        return candy[0];
    }
};
```

### 	605. Can Place Flowers （easy)

```cpp
// ugly version
class Solution {
public:
    bool canPlaceFlowers(vector<int>& flowerbed, int n) {
        int can=0;
        if(accumulate(flowerbed.begin(), flowerbed.end(), 0) == 0)
            return (flowerbed.size()-1)/2+1 >= n;
        int f1,l1;
        for(int i=0;i<flowerbed.size();++i)
            if(flowerbed[i]==1){
                f1=i; 
                can+=i/2;
                break;
            }
        for(int i=flowerbed.size()-1;i>=0;--i)
            if(flowerbed[i]==1){
                l1=i; 
                can+=(flowerbed.size()-1-i)/2;
                break;
            }
        int temp=f1;
        for(int i=f1+1; i<=l1; ++i)
            if(flowerbed[i]==1){
                can+=(i-temp-1-1)/2;
                temp=i;
            }
        return can >= n;
        
        
                
    }
};
```

```cpp
// smart version
class Solution {
public:
    bool canPlaceFlowers(vector<int>& flowerbed, int n) {
        int can=0;
        int len=flowerbed.size();
        if(len<n) return false;
        if(len==0) return true;
        
        vector<int> auxbed(len+2,0);
        copy(flowerbed.begin(), flowerbed.end(), auxbed.begin()+1);
    
        for(int i=1; i<len+1; ++i)
            if(auxbed[i-1]==0 && auxbed[i]==0 && auxbed[i+1]==0 ){
                auxbed[i]=1;
                ++can;
            }
        return can >= n; 
    }
};
// samely smart version
class Solution {
public:
    bool canPlaceFlowers(vector<int>& flowerbed, int n) {
        int can=0;
        int len=flowerbed.size();
        if(len<n) return false;
        if(len==0) return true;
        if(len==1) return flowerbed[0]==0?1>=n:0>=n;
        
        if(flowerbed[0]==0 && flowerbed[1]==0){
            flowerbed[0]=1; ++can;
        }
        if(flowerbed[len-1]==0 && flowerbed[len-2]==0){
            flowerbed[len-1]=1; ++can;
        }
            
        for(int i=1; i<len-1; ++i)
            if(flowerbed[i-1]==0 && flowerbed[i]==0 && flowerbed[i+1]==0 ){
                flowerbed[i]=1;
                ++can;
            }
        return can >= n;
      
    }
};
```

### [435. 无重叠区间](https://leetcode-cn.com/problems/non-overlapping-intervals/)

```CPP
class Solution {
public:
    int eraseOverlapIntervals(vector<vector<int>>& intervals) {
        if(intervals.size()<2) return 0;
        // todo 自定义匿名比较函数-以区间左端点排序
        sort(intervals.begin(), intervals.end(), [](const auto& u, const auto& v) {
            return u[0] < v[0];
        });
        int remove=0;
        int tag=intervals[intervals.size()-1][0];
        for(int i=intervals.size()-2; i>=0; --i){
            if(intervals[i][1]>tag)
                ++remove;
            else
                tag=intervals[i][0];
        }
        return remove;

    }
};
```

```cpp
// version 2
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        if(intervals.empty()) return {};
        sort(intervals.begin(), intervals.end());
        vector<vector<int>> ans;
        ans.push_back(intervals[0]);
        for(int i=1;i<intervals.size();++i){
            int e=ans.size()-1;
            if(intervals[i][0]<=ans[e][1])
                ans[e][1]=max(ans[e][1], intervals[i][1]);
            else
                ans.push_back(intervals[i]);
        }
        return ans;
    }
};

```

### 240. Search a 2D Matrix II
```cpp
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m=matrix.size(), n=matrix[0].size();
        for(int i=m-1,j=0;i>=0&&j<n;)
            if(matrix[i][j]>target)
                --i;
            else if(matrix[i][j]<target)
                ++j;
            else
                return true;
        return false;
    }
};

```


### 452. Arrows to Burst Balloons

```cpp
class Solution {
public:
    int findMinArrowShots(vector<vector<int>>& points) {
        if(points.size()<2) 
            return points.size();
        sort(points.begin(), points.end(), [](const auto& l, const auto& r){ return l[1]<r[1]; });
        int num=1, move=points[0][1];
        for(auto& item:points)
            if(item[0]>move){
                ++num;
                move=item[1];
            }
        return num;
    }
};
```

### 665. Non-decreasing Array

```cpp
class Solution {
public:
    bool checkPossibility(vector<int>& nums) {
        int len=nums.size();
        if(len<3) return true;
        // if(len==3) return nums[0]>nums[1]&&nums[1]>nums[2]?false:true;
        int least=0;
        for(int i=1;i<len && least<2;++i){
            if(nums[i]>=nums[i-1]) continue;
            if(i==len-1) { ++least; continue; }
            if(nums[i+1]<nums[i]) return false;
            if(i==1){ ++least; continue; }
            if(!(nums[i-2]<=nums[i] || nums[i-1]<=nums[i+1])) return false;
            ++least;
        }
        return least<2?true:false;
        
    }
};

```

## 2. 双指针

### 2.1 左右指针

#### 167. Two Sum II - Input Array Is Sorted

```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target) {
        // index solution
        int left=0, right=numbers.size()-1;
        while(left<right)
            numbers[left]+numbers[right]<target ? ++left : --right;
        return vector<int>{left+1, right+1};

        // iterator/pointer solution
        // auto left=numbers.begin(), right=numbers.end()-1;
        // while((*left+*right)!=target)
        //     (*left+*right)>target ? --right : ++left;
        // return vector<int>{left-numbers.begin()+1, right-numbers.begin()+1};


    }
};
```

#### 633. Sum of Square Numbers

```
class Solution {
public:
    bool judgeSquareSum(int c) {
        long long a=0, b=sqrt(c);
        while(a<=b){
            long long sum=a*a+b*b;
            if(sum==c) return true;
            else if(sum>c) --b;
            else ++a;
        }
        return false;
    }
};
```



#### 680. 验证回文字符串 Ⅱ

```cpp
class Solution {
public:
    bool help(const string& s) { //改为匿名函数
        for(int l=0, r=s.size()-1; l<r; ++l, --r)
            if(s[l]!=s[r]) return false;
        return true;
    }
    bool validPalindrome(string s) {
        for(int l=0, r=s.size()-1; l<r; ++l, --r){
            if(s[l]!=s[r]){
                bool dl=help(s.substr(l+1, r-l));
                bool dr=help(s.substr(l, r-l));
                return dl||dr;
            }
        }
        return true;
    }
};
```

#### 88. Merge Sorted Array

```cpp
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int pos=m-- + n-- -1;
        while(m>=0 && n>=0)
            nums1[pos--] = nums1[m]>nums2[n]?nums1[m--]:nums2[n--];
        while(n>=0)
            nums1[pos--]=nums2[n--];
    }
};
```

### 快慢指针

#### 142. Linked List Cycle II

```cpp
class Solution {
public:
    ListNode *detectCycle(ListNode *head) { //version 1
        if(head==nullptr || head->next==nullptr) return nullptr;
        ListNode *slow=head, *fast=head->next; // error1
        while(fast==nullptr || fast->next==nullptr || fast!=slow){
            slow=slow->next;
            fast=fast->next->next;
        }
        if(fast==nullptr || fast->next==nullptr) return nullptr;
        
        // find start of circle
        ListNode *third=head;
        slow=slow->next; // repair error1
        while(third!=slow){
            third=third->next;
            slow=slow->next;
        }
      	return third;
        
        // find length of circle
        int length=0;
        slow=fast;
        do {
            ++length;
            slow=slow->next;
            fast=fast->next->next;
        } while(slow!=fast);
    }
  
    ListNode *detectCycle(ListNode *head) { // right version
      
        // ListNode *slow=head, *fast=head;
        // while(fast!=nullptr&&fast->next!=nullptr && fast!=slow){
        //     slow=slow->next;
        //     fast=fast->next->next;
        // } // invaild loop
      
      
        ListNode *slow=head, *fast=head; //slow和fast需要先分开 do{}while()
        do {
            if(fast==nullptr || fast->next==nullptr) return nullptr;
            slow=slow->next;
            fast=fast->next->next;
        } while(fast!=slow);
        
        // find start of circle
        ListNode *third=head;
        while(third!=slow){
            third=third->next;
            slow=slow->next;
        }
        return third;
    }
};
```

### 滑动窗口

#### 76. Minimum Window Substring

```cpp
class Solution {
public:
    string minWindow(string s, string t) {
        int l=0, r=0, ml=-1, mr=s.size(), cnt=0;
        vector<int> stat(128, 0);
        vector<bool> exist(128, false);

        for(auto c:t){
            ++cnt;
            ++stat[c];
            exist[c]=true;
        }

        for(r=0;r<s.size();++r){
            if(exist[s[r]] && --stat[s[r]]>=0)
                --cnt;
            if(cnt==0 && r-l<mr-ml){
                ml=l; mr=r;
            }
            while(cnt==0 && l<=r){
                if(exist[s[l]] && ++stat[s[l]]>0){
                    ++cnt;
                    if(r-l<mr-ml){
                        ml=l; mr=r;
                    }
                }
                ++l;
            }
        }
        return ml==-1?"":s.substr(ml, mr-ml+1);
    }
};

```
## 3. 二分查找

#### 69. Sqrt(x) (Easy) 
```cpp
class Solution {
public:
    int mySqrt(int x) {
        int l=1, r=x, mid;
        while(l<=r){
            mid=l+(r-l)/2;
            int sqrt=x/mid;
            if(sqrt==mid) return mid;
            else if(sqrt>mid) l=mid+1;
            else r=mid-1;
        }
        return r;
    }
};
```

#### 34. Find First and Last Position of Element in Sorted Array

```cpp
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        if(nums.empty()) 
            return vector<int>{-1, -1};
        int lower=lower_bound(nums, target);
        int upper=upper_bound(nums, target)-1;
        if(upper==-1 || nums[upper]!=target) 
            return vector<int>{-1, -1};
        
        return vector<int>{lower, upper};
    }
    int lower_bound(vector<int>& nums, int target) {
        int l=0, r=nums.size(), mid;
        while(l<r){
            mid=l+(r-l)/2;
            if(nums[mid]>=target)
                r=mid;
            else
                l=mid+1;
        }
        return l;
    }
    int upper_bound(vector<int>& nums, int target) {
        int l=0, r=nums.size(), mid;
        while(l<r){
            mid=l+(r-l)/2;
            if(nums[mid]<=target)
                l=mid+1;
            else
                r=mid;
        }
        return l;
    }
};


```



#### 81. Search in Rotated Sorted Array II

```go
func search(nums []int, target int) bool {
	for l, r := 0, len(nums); l < r; {
		mid := l + (r-l)/2
		if nums[mid] == target {
			return true
		}
		if target < nums[mid] {
			if nums[l] > nums[mid] {
				r = mid
			} else if nums[l] == nums[mid] {
				l++
			} else if nums[l] <= target {
				r = mid
			} else {
				l = mid + 1
			}
		} else {
			if nums[r-1] < nums[mid] {
				l = mid + 1
			} else if nums[r-1] == nums[mid] {
				r--
			} else if nums[r-1] >= target {
				l = mid + 1
			} else {
				r = mid
			}

		}
	}
	return false
}
```



```cpp
class Solution {
public:
    bool search(vector<int>& nums, int target) {
        int l=0, r=nums.size()-1, mid;
        while(l<=r){
            mid=l+(r-l)/2; 
            // 直接二分，讨论mid所在区段
            if(nums[mid]==target) 
                return true;
            if(nums[mid]==nums[l]){
                ++l;    
            } else if(nums[mid]>nums[l]){
                // 直接考虑边界收缩，写出条件
                if(target<nums[mid] && target>=nums[l]) r=mid-1;
                else l=mid+1;
            } else {
                if(nums[mid]<target && target <=nums[r]) l=mid+1;
                else r=mid-1;

            }
        }
        return false; 
    }
};


```

## 4. 排序
### 比较函数
```cpp
#include "stdc++.h"
using namespace std;

template <typename T>
struct CompObj {
    bool operator() (T &l, T&r){
        return l<r;
    }
};

class iint {
    int _i;
public:
    iint(int i):_i(i) {}
    bool operator<(const iint &r) const {
        return this->_i<r._i;
    }

    template <typename T>
    static bool CompFunc(T &l, T &r){
        return r<l;
    }
    friend ostream& operator<<(ostream&, iint&);

};
ostream& operator<<(ostream& os, iint& ii){
    os<<ii._i;
    return os;
}
int main(){
    vector<iint> v{3,2,1,4};
    list<iint> l{6,7,8,5};
    sort(v.begin(), v.end()); // operator <
    for(auto i:v) cout<<i<<' '; cout<<endl; // 1 2 3 4

    sort(v.begin(), v.end(), iint::CompFunc<iint>);// static compfunc -> operator <
    for(auto i:v) cout<<i<<' '; cout<<endl;// 4 3 2 1

    l.sort(CompObj<iint>());// global compobj -> operator <
    for(auto i:l) cout<<i<<' '; cout<<endl;// 5 6 7 8

    l.sort(iint::CompFunc<iint>);// static compfunc -> operator <
    for(auto i:l) cout<<i<<' '; cout<<endl;// 8 7 6 5
}
```

### [215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)

```go
func findKthLargest(nums []int, k int) int {
	// for i, n := 0, len(nums); i < n/2; i++ {
	// 	nums[i], nums[n-1-i] = nums[n-1-i], nums[i]
	// }

	l, r := 0, len(nums)-1
	for l < r {
		for l < r && nums[r] >= nums[0] {
			r--
			// for optimization
			if nums[l+1] <= nums[0] && l+1 < r {
				l++
			}
		}
		for l < r && nums[l] <= nums[0] {
			l++
		}
		if l < r {
			nums[l], nums[r] = nums[r], nums[l]
		}
	}
	nums[r], nums[0] = nums[0], nums[r]

	if l == len(nums)-k {
		return nums[l]
	} else if l < len(nums)-k {
		return findKthLargest(nums[l+1:], k)
	} else {
		return findKthLargest(nums[:l], k-(len(nums)-l))
	}
}

```



```python
class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        if(len(nums)<2):
            return nums[0]
        
        l=0
        r=len(nums)-1
        mid=nums[0]
        
        while(l<r):
            while(l<r and nums[r]>=mid):
                r=r-1
            nums[l]=nums[r]
            while(l<r and nums[l]<=mid):
                l=l+1
            nums[r]=nums[l]
        nums[l]=mid
        
        rlen=len(nums)-l
        if(rlen==k):
            return nums[l]
        if(rlen>k):
            return self.findKthLargest(nums[l+1:], k)
        else:
            return self.findKthLargest(nums[:l], k-rlen)

```
### 347. Top K Frequent Elements

```go
func topKFrequent(nums []int, k int) []int {
	m := make(map[int]int)
	for _, num := range nums {
		m[num]++
	}
	keys := make([]int, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		return m[keys[i]] < m[keys[j]]
	})

	return keys[len(keys)-k:]
}
```

```cpp
347. Top K Frequent Elements
class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> cnt;
        for(int num:nums)
            ++cnt[num];
        int max_cnt=0;
        for(auto i:cnt)
            max_cnt=max(max_cnt, i.second);
        vector<vector<int> > bkt(max_cnt+1);
        for(auto i:cnt)
            bkt[i.second].push_back(i.first);
        vector<int> res;
        for(int i=max_cnt;i>=0;--i){
            for(int num:bkt[i])
                res.push_back(num);
            if(res.size()==k)
                break;
        }
        return res;
    }
};

```
### 75. Sort Colors
```cpp
class Solution {
public:
    void sortColors(vector<int>& nums) {
        int l=0, r=nums.size()-1;
        while(l<r){
            while(l<r&&nums[r]!=0) --r;
            while(l<r&&nums[l]==0) ++l;
            swap(nums[l], nums[r]);   
        }
        r=nums.size()-1;
        while(l<r){
            while(l<r&&nums[r]==2) --r;
            while(l<r&&nums[l]!=2) ++l;
            swap(nums[l], nums[r]);   
        }       
    }
};
```


## 5. 搜索

### DFS

#### 547. Number of Provinces

```cpp
class Solution {
public:
    int findCircleNum(vector<vector<int>>& isConnected) {
        int n=isConnected.size();
        int circle=0;
        for(int i=0; i<n; ++i){
            if(isConnected[i][i])
                ++circle;
            dfs(isConnected, i);
        }
        return circle;
    }
    void dfs(vector<vector<int>>& isConnected, int k){
        isConnected[k][k]=0;
        for(int i=0;i<isConnected[k].size();++i)
            if(isConnected[k][i]==1 && isConnected[i][i]==1)
                dfs(isConnected, i);
        return;
    }
};

```



#### 417. Pacific Atlantic Water Flow

```cpp
class Solution {
public:
    vector<vector<int>> pacificAtlantic(vector<vector<int>>& heights) {
        int m=heights.size(), n=heights[0].size();
        vector<vector<int>> code(m, vector<int>(n, 0));
        for(int i=0;i<n;++i){
            dfs(heights, code, 0, i, 1);
            dfs(heights, code, m-1, i, 2);
        }
        for(int i=1;i<m;++i){
            dfs(heights, code, i, 0, 1);
            dfs(heights, code, m-1-i, n-1, 2);
        }
        
        vector<vector<int>> res;
        for(int i=0;i<m;++i)
            for(int j=0;j<n;++j)
                if(code[i][j]==3)
                    res.push_back(vector<int>{i ,j});
            
        return res; 
    }
    void dfs(vector<vector<int>>& heights, vector<vector<int>>& code, int x, int y, int k){
        
        if(code[x][y]==k || code[x][y]==3) return;
        code[x][y]+=k;
        
        vector<int> dir{0, -1, 0, 1, 0};
        int m=heights.size(), n=heights[0].size();
        for(int i=0;i<4;++i){
            int p=x+dir[i], q=y+dir[i+1];
            if(p>=0 && p<=m-1 && q>=0 && q<=n-1 && 
               heights[x][y]<=heights[p][q])
                dfs(heights, code, p, q, k);
        }
    }
};
```



### 回溯

#### 46. Permutations

```go
func permute(nums []int) [][]int {
	var permutes = [][]int{{}}

	for {
		done := true
		for _, ans := range permutes {
			if len(ans) != len(nums) {
				done = false
				break
			}
		}
		if done {
			break
		}

		var tmps [][]int
		for _, ans := range permutes {
			used := make(map[int]bool)
			for _, num := range ans {
				used[num] = true
			}
			for _, num := range nums {
				if _, ok := used[num]; ok {
					continue
				}
				tmps = append(tmps, append(ans, num))
			}
		}

		permutes = tmps
	}

	return permutes
}
```

```cpp
class Solution {
public:
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> res;
        vector<bool> memo(nums.size(), false);
        vector<int> subres;
        backtracking(nums, res, subres, memo, 0);
        return res;
    }
    void backtracking(vector<int>& nums, vector<vector<int>>& res, vector<int>& subres, vector<bool>& memo, int level){
        
            if(level>=nums.size()){
                res.push_back(subres);
                // return;
            }
            for(int i=0; i<memo.size(); ++i){
                if(!memo[i]){
                    subres.push_back(nums[i]);
                    memo[i]=true;
                    backtracking(nums, res, subres, memo, level+1);
                    memo[i]=false;
                    subres.pop_back();
                }
            }
        }


};
```



#### [77. 组合](https://leetcode.cn/problems/combinations/)

```go
func combine(n int, k int) [][]int {
	var res [][]int
	nums := make([]int, k)
	backtracking(n, k, 1, nums, 0, &res)
	return res
}
func backtracking(n, k, pos int, nums []int, level int, res *[][]int) {
	if level == k {
		temp := make([]int, k)
		copy(temp, nums)
		*res = append(*res, temp)
		return
	}

	for i := pos; i <= n; i++ {
		nums[level] = i
		backtracking(n, k, i+1, nums, level+1, res)
	}
}

```

#### [51. N 皇后](https://leetcode.cn/problems/n-queens/)

```go
func solveNQueens(n int) [][]string {
	var ans [][]string
	cols, ld, rd := make([]int, n), make([]int, 2*n-1), make([]int, 2*n-1)

	board := make([]string, n)
	temp := make([]byte, n)
	for i := range temp {
		temp[i] = byte('.')
	}
	for i := range board {
		board[i] = string(temp)
	}

	backtracking(n, 0, board, cols, ld, rd, &ans)
	return ans
}
func backtracking(n, row int, board []string, cols, ld, rd []int, ans *[][]string) {
	if row == n {
		temp := make([]string, n)
		copy(temp, board)
		*ans = append(*ans, temp)
		return
	}

	for col := 0; col < n; col++ {
		if cols[col] == 0 && ld[row+col] == 0 && rd[row-col+n-1] == 0 {
			temp := []byte(board[row])
			temp[col] = byte('Q')
			board[row] = string(temp)

			cols[col], ld[row+col], rd[row-col+n-1] = 1, 1, 1
			backtracking(n, row+1, board, cols, ld, rd, ans)

			cols[col], ld[row+col], rd[row-col+n-1] = 0, 0, 0
			temp[col] = byte('.')
			board[row] = string(temp)

		}
	}
}
```



### BFS

#### 934. Shortest Bridge

```go
func shortestBridge(grid [][]int) (step int) {
	type seat struct {
		x, y int
	}
	q := make([]seat, 0)
	for i, row := range grid {
		for j, unit := range row {
			if unit == 1 {
				q = append(q, seat{i, j})
				break
			}
		}
		if len(q) != 0 {
			break
		}
	}

	if len(q) == 0 {
		return
	}

	dirs := []int{-1, 0, 1, 0, -1}
	m, n := len(grid), len(grid[0])

	// find island
	for i := 0; i < len(q); i++ {
		p := q[i]

		grid[p.x][p.y] = 2
		for i := 0; i < 4; i++ {
			if x, y := p.x+dirs[i], p.y+dirs[i+1]; x >= 0 && x < m &&
				y >= 0 && y < n && grid[x][y] == 1 {
				grid[x][y] = 2
				q = append(q, seat{x, y})
			}
		}
	}

	// expand island by BFS
	q2 := make([]seat, 0)
	for len(q) != 0 {
		p := q[0]
		q = q[1:]

		for i := 0; i < 4; i++ {
			if x, y := p.x+dirs[i], p.y+dirs[i+1]; x >= 0 && x < m && y >= 0 && y < n {
				if grid[x][y] == 0 {
					grid[x][y] = 2
					q2 = append(q2, seat{x, y})
				} else if grid[x][y] == 1 {
					return
				}
			}
		}

		if len(q) == 0 {
			step++
			q, q2 = q2, make([]seat, 0)
		}
	}

	return
}

```





```cpp
class Solution {
public:
    int shortestBridge(vector<vector<int>>& grid) {
        int m=grid.size()-1, n=grid[0].size()-1;
        queue<pair<int,int>> points;
        bool flag=true;
        for(int i=0;i<=m&&flag;++i)
            for(int j=0;j<=n;++j)
                if(grid[i][j]){
                    flag=false;
                    dfs(points, grid, {i,j}, {m,n});
                    break;
                }
        // bfs
        int step=0;
        while(!points.empty()){
            int np=points.size();
            while(np--){
                pair<int, int> coor=points.front();
                points.pop();
                vector<int> dir={-1, 0, 1, 0, -1};
                for(int i=0;i<4;++i){
                    int x=coor.first+dir[i], y=coor.second+dir[i+1];
                    if(x>=0&&x<=m&&y>=0&&y<=n) {
                        if(grid[x][y]==1) return step;
                        if(grid[x][y]==0){
                            grid[x][y]=2;
                            points.push({x, y});
                        }
                        
                    }
                }
            }
            ++step;
        }
        return 0;
    }
    void dfs(queue<pair<int,int>>& points, vector<vector<int>>& grid, pair<int,int> coor, pair<int, int> bound) {
        grid[coor.first][coor.second]=2;
        points.push(coor);
        
        vector<int> dir{-1, 0, 1, 0, -1};
        for(int i=0;i<4;++i){
            int x=coor.first+dir[i], y=coor.second+dir[i+1];
            if(x>=0&&x<=bound.first&&y>=0&&y<=bound.second&&grid[x][y]==1){
                dfs(points, grid, {x, y}, bound);
            }
        }
    }
};


```

## 6. DP



[413. 等差数列划分](https://leetcode.cn/problems/arithmetic-slices/)

```go
func numberOfArithmeticSlices(nums []int) int {
	n := len(nums)
	if n < 3 {
		return 0
	}

	dp := make([]int, len(nums))
	for i := 2; i < n; i++ {
		if nums[i]+nums[i-2] == 2*nums[i-1] {
			dp[i] = dp[i-1] + 1 // 1 means new sub sequence 1-2,1-1,i
		}
	}

	res := 0
	for _, num := range dp {
		res += num
	}

	return res
}

func numberOfArithmeticSlices2(nums []int) int {
	if len(nums) < 3 {
		return 0
	}

	tail := make([]int, len(nums))
	tailN := make([]int, len(nums))
	// if nums[0]+nums[2] == 2*nums[1] {
	// 	tail[2] = 1
	// 	tailN[2] = 3
	// }

	res := 0
	for i := 2; i < len(nums); i++ {
		if nums[i]+nums[i-2] == 2*nums[i-1] {
			if tailN[i-1] == 0 {
				tail[i] = 1
				tailN[i] = 3
			} else {
				tail[i] = tailN[i-1] - 1 + tail[i-1]
				tailN[i] = tailN[i-1] + 1
			}
		} else {
			res += tail[i-1]
		}

	}

	return res + tail[len(nums)-1]
}
```



### 二维

#### 542. 01 Matrix

```go
func updateMatrix(mat [][]int) [][]int {
	m := len(mat)
	if m == 0 {
		return nil
	}
	n := len(mat[0])

	for i := range mat {
		for j := range mat[i] {
			if mat[i][j] == 1 {
				mat[i][j] = m + n // max distance value
			}
		}
	}

	min := func(a, b int) int {
		if a < b {
			return a
		}
		return b
	}
	// update for right and down direction
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			dirs := []int{0, 1, 0}
			for k := range []int{0, 1} {
				x, y := i+dirs[k], j+dirs[k+1]
				if x >= 0 && x < m && y >= 0 && y < n {
					mat[x][y] = min(mat[x][y], mat[i][j]+1)
				}
			}
		}
	}
	// update for left and up direction from the bottom right point
	for i := m - 1; i >= 0; i-- {
		for j := n - 1; j >= 0; j-- {
			dirs := []int{-1, 0, -1}
			for k := range []int{0, 1} {
				x, y := i+dirs[k], j+dirs[k+1]
				if x >= 0 && x < m && y >= 0 && y < n {
					mat[x][y] = min(mat[x][y], mat[i][j]+1)
				}
			}
		}
	}

	return mat
}
```
```CPP
class Solution {
public:
    vector<vector<int>> updateMatrix(vector<vector<int>>& mat) {
        int m=mat.size(), n=mat[0].size();
        vector<vector<int>> res(m, vector<int>(n, INT_MAX-1));
        for(int i=0;i<m;++i)
            for(int j=0;j<n;++j)
                if(mat[i][j]==0)
                    res[i][j]=0;
                else {
                    if(j>0)
                        res[i][j]=min(res[i][j], res[i][j-1]+1);
                    if(i>0)
                        res[i][j]=min(res[i][j], res[i-1][j]+1);
                }
        // back to left top
        for(int i=m-1;i>=0;--i)
            for(int j=n-1;j>=0;--j)
                if(mat[i][j]==0) 
                    res[i][j]=0;
                else {
                    if(j<n-1)
                        res[i][j]=min(res[i][j], res[i][j+1]+1);
                    if(i<m-1)
                        res[i][j]=min(res[i][j], res[i+1][j]+1);
                }
                            
        return res;
    }
};
```

#### 221. Maximal Square

```cpp
class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        if(matrix.size()==0||matrix[0].size()==0) return 0;
        int max_edge=0;
        int m=matrix.size(), n=matrix[0].size();
        vector<vector<int>> dp(m, vector<int>(n, 0));
        for(int i=0;i<m;++i)
            for(int j=0;j<n;++j){
                if(i==0 || j==0)
                    dp[i][j]=matrix[i][j]-'0';
                else if(matrix[i][j]=='1')
                    dp[i][j]=min(min(dp[i-1][j], dp[i][j-1]), dp[i-1][j-1])+1;
                
                max_edge=max(max_edge, dp[i][j]);
            }
        return max_edge*max_edge;

        
    }
};
```

### 分割

#### 91. Decode Ways

```cpp
class Solution {
public:
    int numDecodings(string s) {
        int n=s.size();
        if(n==1&&s[0]=='0') return 0;
        if(n<2) return n;
        vector<int> dp(n+1);
        dp[0]=1;
        dp[1]=1;
        int pre, cur;
        for(int i=2;i<=n;++i){
            pre=s[i-2]-'0';
            cur=s[i-1]-'0';
            int temp=pre*10+cur;
            if(cur==0){
                if(pre==1||pre==2)
                    dp[i]=dp[i-2];
                else
                    return 0;
            }
            else if(i-2==0&&pre==0)
                return 0;
            else if(temp>26||temp<10)
                dp[i]=dp[i-1];
            else 
                dp[i]=dp[i-1]+dp[i-2];
                
        }
        return dp[n];  
    }
};


```
### 子序列
#### 300. Longest Increasing Subsequence
一维DP
```cpp
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int n=nums.size();
        
        vector<int> dp(n, 1);
        int max_len=0;
        for(int i=0;i<n;++i){
            for(int j=i-1;j>=0;--j){
                if(nums[i]>nums[j])
                    dp[i]=max(dp[i], dp[j]+1);
            }
            max_len=max(max_len, dp[i]);
        }
        return max_len;
        
    }
};
```
#### 1143. Longest Common Subsequence
二维DP
```cpp
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        int m=text1.size(), n=text2.size();
        vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
        for(int i=0;i<m;++i){
            for(int j=0;j<n;++j){
                if(text1[i]==text2[j])
                    dp[i+1][j+1]=dp[i][j]+1;
                else 
                    dp[i+1][j+1]=max(dp[i][j+1], dp[i+1][j]);
            }
        }
        return dp[m][n];
    }
};
```
### 编辑字符串
#### 72. Edit Distance
```cpp
class Solution {
public:
    int minDistance(string word1, string word2) {
        int m=word1.size(), n=word2.size();
        vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
        for(int i=0;i<=m;++i)
            dp[i][0]=i;
        for(int i=0;i<=n;++i)
            dp[0][i]=i;
        for(int i=1;i<=m;++i)
            for(int j=1;j<=n;++j){ // 打表 考虑所有修改情况
                dp[i][j] = word1[i-1]==word2[j-1] ? dp[i-1][j-1] : 
                    min(dp[i-1][j-1], min(dp[i-1][j], dp[i][j-1]))+1;
            }
        return dp[m][n];
    }
};
```
#### 650. 2 Keys Keyboard
```cpp
class Solution {
public:
    int minSteps(int n) {
        vector<int> dp(n+1, 0);
        for(int i=2;i<=n;++i){
            dp[i]=i;
            for(int j=2;j*j<=i;++j)
                if(i%j==0) // 倍数/因数类DP
                    dp[i]=dp[j]+dp[i/j];
        }
        return dp[n];
    }
};
```
#### 10. Regular Expression Matching

```cpp
class Solution {
public:
    bool isMatch(string s, string p) {
        int m=s.size(), n=p.size();
        vector<vector<bool>> dp(m+1, vector<bool>(n+1, false));
        for(int i=2;i<=n&&p[i-1]=='*';i+=2){
            dp[0][i]=true;
        }
        dp[0][0]=true;
        dp[1][1]=(p[0]=='.'||p[0]==s[0])?true:false;
        
        for(int i=1;i<=m;++i)
            for(int j=2;j<=n;++j){
                if(p[j-1]=='*'){ //考虑所有情况 1消耗 2同归 3直接走
                    if(p[j-2]==s[i-1]||p[j-2]=='.')
                        dp[i][j]=dp[i-1][j-2]||dp[i-1][j];
                    dp[i][j]=dp[i][j]||dp[i][j-2];
                }
                else if(p[j-1]=='.' || p[j-1]==s[i-1])
                    dp[i][j]=dp[i-1][j-1];
                else
                    dp[i][j]=false;
            }
        return dp[m][n];     
    }
};
```
## 7. 分治
### 932. 漂亮数组
```cpp
class Solution {
public:
    vector<int> beautifulArray(int n) {
        if(n==1) return {1};
        vector<int> ans;
        vector<int> temp=beautifulArray((n+1)/2);
        for(auto item:temp){
            ans.push_back(2*item-1);
        }
        for(auto item:temp){
            if(2*item<=n)
                ans.push_back(2*item);
        }
        return ans;
    }
};
```
### 241. Different Ways to Add Parentheses
```cpp
class Solution {
public:
    vector<int> diffWaysToCompute(string input) {
        int n=input.size();
        vector<int> ways;
        for(int i=0;i<n;++i){
            char punc=input[i];
            if(punc>='0' && punc<='9') continue;
            vector<int> left=diffWaysToCompute(input.substr(0, i));
            vector<int> right=diffWaysToCompute(input.substr(i+1));
            for(auto l:left)
                for(auto r:right){
                    switch(punc){
                    case '+':ways.push_back(l+r);break;
                    case '-':ways.push_back(l-r);break;
                    case '*':ways.push_back(l*r);break;
                    }
                }
        }
        if(ways.empty()) ways.push_back(stoi(input));
        return ways;
    }
    
};
```



## 8. 数学

### 415. Add Strings

```cpp
class Solution {
public:
    string addStrings(string num1, string num2) {
        int n1=num1.size(), n2=num2.size();
        int n=max(n1, n2)+1;
        vector<int> add(n, 0);
        for(int i=n1-1;i>=0;--i)
            add[--n]+=num1[i]-'0';
        n=max(n1, n2)+1;
        for(int i=n2-1;i>=0;--i)
            add[--n]+=num2[i]-'0';
        
        string ans;
        for(int i=add.size()-1;i>=0;--i){
            cout<<add[i]<<' ';
            ans=to_string(add[i]%10)+ans;
            if(add[i]>=10)
                ++add[i-1];
        }
        if(ans[0]=='0')
            return ans.substr(1);
        return ans;
    }
};
```

### 172. 阶乘后的零

```cpp
class Solution {
public:
    int trailingZeroes(int n) {
        int cnt=0;
        while(n>0){
            n /= 5;
            cnt+=n;
        }
        return cnt;
    }
};
```

### [233. 数字 1 的个数](https://leetcode-cn.com/problems/number-of-digit-one/)

[题解公式](https://leetcode-cn.com/problems/number-of-digit-one/solution/shu-zi-1-de-ge-shu-by-leetcode-solution-zopq/)

```cpp
class Solution {
public:
    int countDigitOne(int n) {
        // mulk 表示 10^k
        // 在下面的代码中，可以发现 k 并没有被直接使用到（都是使用 10^k）
        // 但为了让代码看起来更加直观，这里保留了 k
        long long mulk = 1;
        int ans = 0;
        for (int k = 0; n >= mulk; ++k) {
            ans += (n / (mulk * 10)) * mulk + min(max(n % (mulk * 10) - mulk + 1, 0LL), mulk);
            mulk *= 10;
        }
        return ans;
    }
};
```

## 9. 数据结构

### 数组

#### 769. Max Chunks To Make Sorted

```cpp
class Solution {
public:
    int maxChunksToSorted(vector<int>& arr) {
        int cur_max=0, chunk=0;
        for(int i=0;i<arr.size();++i){
            cur_max=max(cur_max,arr[i]);
            if(cur_max==i)
                ++chunk;
        }
        return chunk;
    }
};
```
### 栈和队列

#### 155. Min Stack

```cpp
class MinStack {
    stack<int> s, min_s;
public:
    MinStack() {
        
    }
    
    void push(int val) {
        s.push(val);
        if(min_s.empty() || val<=min_s.top())
            min_s.push(val);
    }
    
    void pop() {
        if(s.empty()) return;
        if(s.top()==min_s.top())
            min_s.pop();
        s.pop();
    }
    
    int top() {
        return s.top();
    }
    
    int getMin() {
        return min_s.top();
    }
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(val);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->getMin();
 */
```
#### 739. Daily Temperatures

```cpp

class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        auto n=temperatures.size();
        if(n==0) return {};
        vector<int> ans(n);
        stack<int> s;
        for(int i=0;i<n;++i){
            while(!s.empty()&&temperatures[i]>temperatures[s.top()]){
                    ans[s.top()]=i-s.top();
                    s.pop();
                }
            s.push(i);
        }
        return ans;
    }
};
```
### 优先队列

```cpp
class priority_queue(){
    vector<int> heap(1); // heap[0]置空 //heap[1]=pq.top()
public:
    void swim(int pos){
        while(pos>1 && heap[pos]>heap[pos/2]){
            swap(heap[pos], heap[pos/2]);
            pos/=2;
        }
    }
    void sink(int pos){
        int N=heap.size()-1;
        while(2*pos<=N){
            int lc=2*pos, rc=lc+1;
            if(rc<=N&&heap[lc]<heap[rc]) lc=rc;
            if(heap[lc]<=heap[pos]) break;
            swap(heap[lc], heap[pos]);
            pos=lc;
        }
    }
}
```

#### 23. Merge k Sorted Lists

```cpp
// REDIFINATION LISTNODE
// struct ListNode {
//     int val;
//     ListNode *next;
//     ListNode() : val(0), next(nullptr) {}
//     ListNode(int x) : val(x), next(nullptr) {}
//     ListNode(int x, ListNode *next) : val(x), next(next) {}
//     bool operator < (ListNode* l, ListNode* r) { return l->val<r->val; }
// };

// struct CompObj {
//     bool operator() (ListNode* l, ListNode* r) { return l->val<r->val; }
// };
// auto CompFun=[](ListNode* l, ListNode* r) { return l->val<r->val; };

// 优先级队列
// class Solution {
// public:
//     ListNode* mergeKLists(vector<ListNode*>& lists) {
//         priority_queue<ListNode*, vector<ListNode*>, decltype(CompFun)> pq(CompFun);
//         for(auto ls:lists){
//             while(ls!=nullptr){
//                 pq.push(ls);
//                 ls=ls->next;
//             }
//         }
//         ListNode* head=nullptr;
//         while(!pq.empty()){
//             auto max_pq=pq.top();
//             pq.pop();
//             max_pq->next=head;
//             head=max_pq;
//         }
//         return head;
//     }
// };

// 分治
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode* head = nullptr;
        ListNode** pre = &head, **node;
        for(node = NULL;list1 && list2; *node = (*node)->next)
        {
            node = (list1->val < list2->val) ? &list1 : &list2;
            *pre = *node;
            pre = &(*node)->next;
        }
        *pre = (ListNode *)((uintptr_t) list1 | (uintptr_t)list2);
        return head;
    }
    ListNode* mergeList(vector<ListNode*>& lists, int start, int end)
    {
        if(start==end) return lists[start];
        int mid = (end - start)/2 + start;
        ListNode* left = mergeList(lists, start, mid);
        ListNode* right = mergeList(lists, mid+1, end);
        
        return mergeTwoLists(left, right);
    }
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        int size = lists.size();
        if(size == 0)
            return NULL;
        else
            return mergeList(lists, 0, size - 1);
    }
};
```
#### 218. 天际线问题

```cpp
class Solution1 {
public:
    vector<vector<int>> getSkyline(vector<vector<int>>& buildings) {
        vector<pair<int,int>> od;
        for(auto b:buildings){
            od.push_back({b[0],-b[2]});
            od.push_back({b[1],b[2]});
        }
        sort(od.begin(), od.end());
        priority_queue<int> pq;
        priority_queue<int> rm;
        vector<vector<int>> ans;
        
        int prev=0, n=buildings.size();
        pq.push(0);
        for(auto p:od){
            if(p.second<0)
                pq.push(-p.second);
            else if(p.second>0){
                // cpp unsupport pq.erase(elem)
                rm.push(p.second);
            }
            while(!rm.empty() && rm.top()==pq.top()){
                    rm.pop(); pq.pop();
            }
            int cur=pq.top();
            if(prev!=cur){
                ans.push_back({p.first, cur});
                prev=cur;
            }
        }
        return ans;
    }
    
};
```
```cpp
vector<vector<int>> getSkyline(vector<vector<int>>& buildings) { 
    vector<vector<int>> ans;
    priority_queue<pair<int, int>> max_heap; // <高度, 右端>
    int i = 0, len = buildings.size();
    int cur_x, cur_h;
    while (i < len || !max_heap.empty()) {
        if (max_heap.empty() || i < len && buildings[i][0] <= max_heap.top().second) {
            cur_x = buildings[i][0];
            while (i < len && cur_x == buildings[i][0]) {
                max_heap.emplace(buildings[i][2], buildings[i][1]);
                ++i; 
            }
        } else {
            cur_x = max_heap.top().second;
            while (!max_heap.empty() && cur_x >= max_heap.top().second) {
                max_heap.pop();
            }
        }
        cur_h = (max_heap.empty()) ? 0 : max_heap.top().first;
        if (ans.empty() || cur_h != ans.back()[1]) {
            ans.push_back({cur_x, cur_h});
        }
    }
    return ans; 
}
```
#### 239. Sliding Window Maximum

- version1
```cpp
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        deque<int> dq;
        vector<int> ans;
        for(int i=0;i<nums.size();++i){
            while(!(dq.empty()||nums[dq.back()]>nums[i])){
                dq.pop_back();
            }
            dq.push_back(i);
            if(dq.front()<=i-k)
                dq.pop_front();
            if(i>=k-1)
                ans.push_back(nums[dq.front()]);
        }
        return ans;
    }
};
```
- version2
```cpp
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> ks;
        vector<int> ans;
        int front=0;
        for(int i=0;i<nums.size();++i){
            while(ks.size()>front && nums[ks.back()]<=nums[i]){
                ks.pop_back();
            }
            ks.push_back(i);
            if(ks[front]<=i-k)
                ++front;
            if(i>=k-1)
                ans.push_back(nums[ks[front]]);
        }
        return ans;
    }
};
```

### 多重集合和映射

#### 332. Reconstruct Itinerary

```cpp
class Solution {
public:
    vector<string> findItinerary(vector<vector<string>>& tickets) {
        vector<string> ans;
        if (tickets.empty()) {
            return ans; }
        unordered_map<string, priority_queue<string,vector<string>, greater<string>>> hash; 
        // unordered_map<string, multiset<string>> hash;         // 可以使用multiset
        for (const auto & ticket: tickets) {
           hash[ticket[0]].push(ticket[1]);
        }
        stack<string> s;
        s.push("JFK");
        while (!s.empty()) {
           string next = s.top();
           if (hash[next].empty()) {
               ans.push_back(next);
               s.pop();
           } else {
               s.push(hash[next].top());
               hash[next].pop();
           }
        }
        reverse(ans.begin(), ans.end());
        return ans;
    }
};
```
### 前缀和与积分图
#### 560. Subarray Sum Equals K
```cpp
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        unordered_map<int, int> mapping;
        int cnt=0, psum=0;
        mapping[0]=1;
        for(auto num:nums){
            psum+=num;
            cnt+=mapping[psum-k];
            ++mapping[psum];
        }
        return cnt;
    }
};
```

## 10. 字符串

### class String
```cpp
#include<iostream>
#include<vector>
using namespace std;
class String {
    int len;
    char *str;
public:
    String():len(0), str(nullptr) {}
    String(int len, const char *str):len(0), str(nullptr){
        if(len<=0) return;
        this->len=len;
        this->str=new char[len];
        for(int i=0;i<len;++i)
            *(this->str+i)=*(str+i);
    }
    String(const String& oth):len(oth.len), str(nullptr){
        if(len<=0) return;
        str=new char[len];
        for(int i=0;i<len;++i)
            *(str+i)=*(oth.str+i);
    }

    String& operator=(const String& oth){
        if(&oth==this) return *this;
        delete [] str;
        str=nullptr;
        len=oth.len;
        str=new char[len];
        for(int i=0;i<len;++i)
            *(str+i)=*(oth.str+i);
        return *this;
    }
    ~String(){
        cout<<"deconstructing ";
        this->print();
        delete [] str;
        len=0;
        str=nullptr;

    }
    void print() const {
        if(len==0)
            cout<<'"'<<'"'<<endl;
        else {
            cout<<'"';
            for(int i=0;i<len;++i)
                cout<<*(str+i);
            cout<<'"'<<endl;
        }
    }
};
int main(){
    char str[]="abc";
    String s1;
    String s2(3, str);
    String s3(s2);
    s1.print();
    s2.print();
    s3.print();
    s1=s2;
    s1.print();
    auto *p=new int [10];
    cout<<p[5]<<endl;
    delete p;

    return 0;

}
```



#### 205. Isomorphic Strings

```cpp
class Solution {
public:
    bool isIsomorphic(string s, string t) {
        int n=s.size();
        if(n!=t.size()) return false;
        vector<int> sm(128, 0);
        vector<int> pm(128, 0);
        for(int i=0;i<n;++i){
            if(sm[s[i]]!=pm[t[i]]) 
                return false;
            sm[s[i]]=pm[t[i]]=i+1;
        }
        return true;
    }
};
```



```cpp
class Solution {
public:
    bool isIsomorphic(string s, string t) {
        int n=s.size();
        if(n!=t.size()) return false;
        vector<int> sm(128, 0);
        vector<int> pm(128, 0);
        for(int i=0;i<n;++i){
            if(sm[s[i]]!=pm[t[i]]) 
                return false;
            sm[s[i]]=pm[t[i]]=i+1;
        }
        return true;
    }
};
```

## 11. 链表

### 206. Reverse Linked List

```cpp
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* prev=nullptr;
        while(head!=nullptr){
            ListNode* next=head->next;
            head->next=prev;
            prev=head;
            head=next;
        }
        return prev;
    }
    // ListNode* reverseList(ListNode* head, ListNode* prev=nullptr) {
    //     if(head==nullptr) return prev;
    //     ListNode* next=head->next;
    //     head->next=prev;
    //     return reverseList(next, head);
    // }
};
```

### 21. Merge Two Sorted Lists

```cpp
class Solution {
public:
    // ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
    //     ListNode *tail=new ListNode(), *head=tail;
    //     while(list1 || list2){
    //         if(!list1 || list2&&(list1->val>list2->val)){
    //             tail->next=list2;
    //             list2=list2->next;
    //         }
    //         else {
    //             tail->next=list1;
    //             list1=list1->next;
    //         }
    //         tail=tail->next;
    //     }
    //     return head->next;
    // }
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        if(!list1)
            return list2;
        if(!list2)
            return list1;
        if(list1->val > list2->val){
            list2->next=mergeTwoLists(list1, list2->next);
            return list2;
        }
        list1->next=mergeTwoLists(list1->next, list2);
        return list1;
    }
    
};
```



### 234. Palindrome Linked List

```cpp
class Solution {
public:
    bool isPalindrome(ListNode* head) {
        // list转vector更快 学会选择最优数据结构
        ListNode *fast=head, *slow=head;
        while(fast && fast->next){
            fast=fast->next->next;
            slow=slow->next;
        }
        
        ListNode *rhead=nullptr, *next;
        while(slow){
            next=slow->next;
            slow->next=rhead;
            rhead=slow;
            slow=next;
            
        }
        while(head && rhead){
            if(head->val!=rhead->val)
                return false;
            head=head->next;
            rhead=rhead->next;
        }
        return true;
    }
};
```

### 24. Swap Nodes in Pairs

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        // recursive
        if(!head||!head->next) return head;
        ListNode *next=head->next;
        head->next=swapPairs(next->next);
        next->next=head;
        return next;
        // loop
//         ListNode *cur=head, *next, *tail=new ListNode();
//         if(head&&head->next)
//             head=head->next;
//         while(cur && cur->next){
//             next=cur->next;
//             tail->next=next; 
//             cur->next=next->next;
//             next->next=cur;
            
//             tail=cur;
//             cur=cur->next;
//         }
//         return head;
    }
};
```
### 148. 排序链表

指针快排方式的链表排序，注意规避最坏情况。
建议链表排序使用归并排序
```cpp
class Solution {
public:
    ListNode* sortList(ListNode *head) {
        ListNode *fake=new ListNode();
        fake->next=head;
        return sortList(fake, nullptr); // 给链表添加一个辅助头和辅助尾，即左右都不包含，开区间排序
    }
    ListNode* sortList(ListNode *fake, ListNode *tail) {
        // if(!fake) return fake; // the fake never fakes
        ListNode *head=fake->next;
        if(head==tail || head->next==tail) return head;

        // 交换链表中点和首点，避免增序最坏情况
        ListNode *fast=head, *slow=head, *tmp;
        while(fast!=tail && fast->next!=tail){
            fast=fast->next->next;
            slow=slow->next;
        }
        if(slow->next!=tail){ // 交换head和slow->next
            tmp=slow->next->next;
            fake->next=slow->next;
            slow->next->next=head->next;
            slow->next=head;
            head->next=tmp;

            head=fake->next; // head重定向
        }

        ListNode *pivot=head, *cur=head, *prev;
        while(cur->next!=tail){
            prev=cur;
            while (cur->next!=tail && cur->next->val < pivot->val){ // 连续
                cur=cur->next;
            }
            if(prev!=cur){ // 互换
                tmp=cur->next;
                cur->next=head;
                head=prev->next;
                prev->next=tmp;
                fake->next=head;
                cur=prev; // 只剔除
            } else {
                cur=cur->next; // 只前进
            }
        }
        head=sortList(fake, pivot); // 开区间
        tmp=sortList(pivot, tail); // 开区间
        return head?head:pivot;
    }
};
```



## 12. 树

### 104. Maximum Depth of Binary Tree

```cpp
class Solution {
public:
    int maxDepth(TreeNode* root) {
        return root?1+max(maxDepth(root->left), maxDepth(root->right)):0;
    }
};
```

### 110. 平衡二叉树
```cpp
class Solution {
public:
    bool isBalanced(TreeNode* root) {
        return getMaxDepth(root)!=-1; // 求深度可以判断是否平衡
    }
    int getMaxDepth(TreeNode *root){
        if(!root) return 0;
        int l=getMaxDepth(root->left);
        if(l==-1) return -1;
        int r=getMaxDepth(root->right);
        if(r==-1) return -1;
        if(abs(l-r)>1) return -1;
        return max(l, r)+1;
    }
};
```
### 543. Diameter of Binary Tree
```cpp
class Solution {
public:
    int diameterOfBinaryTree(TreeNode* root) {
        int diam=0;
        helper(root, diam);
        return diam;
    }
    int helper(TreeNode *root, int &diam){
        if(!root) return 0;
        int l=helper(root->left, diam);
        int r=helper(root->right, diam);
        diam=max(diam, l+r);
        return max(l, r)+1;
    }
};
```
### 437. Path Sum III
```cpp
class Solution {
public:
    int pathSum(TreeNode* root, int sum) {
        if(!root) return 0;
        return pathSum(root->right, sum) + pathSum(root->left, sum) +
            pathWithRoot(root, sum);
    }
    int pathWithRoot(TreeNode *root, int sum){
        if(!root) return 0;
        sum-=root->val;
        int cnt=sum==0? 1: 0;
        cnt+=pathWithRoot(root->left, sum);
        cnt+=pathWithRoot(root->right, sum);
        return cnt;
    }
};
```
### 1110. Delete Nodes And Return Forest
```cpp
class Solution {
public:
    // 主函数
    vector<TreeNode*> delNodes(TreeNode* root, vector<int>& to_delete) {
        vector<TreeNode*> forest;
        unordered_set<int> dict(to_delete.begin(), to_delete.end());
        root = helper(root, dict, forest);
        if (root) {
           forest.push_back(root);
        }
        return forest;
    }
    // 辅函数
    TreeNode* helper(TreeNode* root, unordered_set<int> & dict, vector<TreeNode*> & forest) {
        if (!root) {
           return root;
        }
        root->left = helper(root->left, dict, forest);
        root->right = helper(root->right, dict, forest);
        if (dict.count(root->val)) {
           if (root->left) {
               forest.push_back(root->left);
           }
           if (root->right) {
               forest.push_back(root->right);
           }
            root = NULL; 
        }
        return root;
    }
};
```

```cpp
class Solution {
public:
    vector<TreeNode*> delNodes(TreeNode* root, vector<int>& to_delete) {
        unordered_set<int> del_set(to_delete.begin(), to_delete.end());
        vector<TreeNode*> ans;
        helper(root, true, del_set, ans);
        return ans;
    }
    TreeNode* helper(TreeNode *root, bool p_is_null, unordered_set<int> &del_set, vector<TreeNode*> &ans) {
        if(!root) return root;
        TreeNode *res=root;
        if(del_set.count(root->val)) {
            res=nullptr;
            p_is_null= true;
        }   
        else if(p_is_null) {
            ans.push_back(root);
            p_is_null=false;
        }

        root->left = helper(root->left, p_is_null, del_set, ans);
        root->right = helper(root->right, p_is_null, del_set, ans);
        return res;
    }
};

```
105. Construct Binary Tree from Preorder and Inorder Traversal
```cpp
class Solution {
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        if(preorder.empty()) return nullptr;
        unordered_map<int, int> hash;
        for(int i=0;i<inorder.size();++i){
            hash[inorder[i]]=i;
        }

        return helper(preorder, 0, hash, 0, inorder.size()-1);

    }
    TreeNode* helper(vector<int> &preorder, int rt_i, unordered_map<int, int> &hash, int st_i, int end_i){
        if(st_i>end_i) return nullptr;
        int pre_rt=preorder[rt_i], mid_i=hash[pre_rt];
        TreeNode *root=new TreeNode(pre_rt);
        root->left=helper(preorder, rt_i+1, hash, st_i, mid_i-1);
        root->right=helper(preorder, rt_i+mid_i-st_i+1, hash, mid_i+1, end_i);
        return root;
    }
};
```
144. Binary Tree Preorder Traversal
```cpp
class Solution {
    vector<int> ans;
public:
    const vector<int>& preorderTraversal(TreeNode* root) {
        if(!root) return ans;
        ans.push_back(root->val);
        preorderTraversal(root->left);
        preorderTraversal(root->right);
        return ans;
    }
};
```
669. Trim a Binary Search Tree

```cpp
class Solution {
public:
    TreeNode* trimBST(TreeNode* root, int low, int high) {
        if(!root) return nullptr;
        if(root->val<low)
            return trimBST(root->right, low, high);
        if(root->val>high)
            return trimBST(root->left, low, high);
        root->left=trimBST(root->left, low, high);
        root->right=trimBST(root->right, low, high);
        return root;
    }
};
```
### 144. Binary Tree Preorder Traversal
class Solution {
    vector<int> ans;
public:
    const vector<int>& preorderTraversal(TreeNode* root) {
        if(!root) return ans;
        stack<TreeNode*> s;
        s.push(root);
        while(!s.empty()){
            root=s.top();
            if(root){
                ans.push_back(root->val);
                s.pop();
                s.push(root->right);
                s.push(root->left);
            } else {
                s.pop();
            }
        }
        return ans;
    }
};
### 二叉查找树
```cpp
template<typename T>
class BST {
    struct TreeNode {
        T data;
        TreeNode *left;
        TreeNode *right;
    };
    TreeNode *root;

    TreeNode *insert(TreeNode *node, T x) {
        if (!node) {
            node = new TreeNode;
            node->data = x;
            node->left = node->right = nullptr;
            return node;
        }
        if (x < node->data) node->left = insert(node->left, x);
        else node->right = insert(node->right, x);
        return node;

    }

    TreeNode *remove(TreeNode *node, T x) {
        TreeNode *temp;
        if (!node) return node;
        if (x < node->data) node->left = remove(node->left, x);
        else if (x > node->data) node->right = remove(node->right, x);
        else if (node->left && node->right) {
            tmep = findMin(root->right);
            node->data = temp->data;
            node->right = remove(node->right, node->data);
        } else {
            temp = node;
            if (!node->left) node = node->right;
            else node = node->left;
            delete temp;
        }
        return node;
    }

    TreeNode *find(TreeNode *node, T x) {
        if (node == nullptr) return nullptr;
        if (x < t->data) return find(node->left, x);
        if (x > t->data) return find(node->right, x);
        return node;
    }

    TreeNode *findMin(TreeNode *node) {
        if (!root || !root->left) return root;
        return findMin(root->left);
    }

    TreeNode *findMax(TreeNode *node) {
        if (!root || !root->right) return root;
        return findMax(root->right);
    }

    TreeNode *makeEmpty(TreeNode *node) {
        if (!root) return nullptr;
        makeEmpty(node->left);
        makeEmpty(node->right);
        delete node;
        return nullptr;
    }

public:
    BST() : root(nullptr) {}

    ~BST() {
        root = makeEmpty(root);
    }

    void insert(T x) {
        insert(root, x);
    }

    void remove(T x) {
        remove(root, x);
    }
};
```
## 13. 图
### 785. Is Graph Bipartite?
```cpp
class Solution {
public:
    bool isBipartite(vector<vector<int>>& graph) {
        int n=graph.size(), cur;
        vector<int> color(n, 0);
        queue<int> q;
        for(int i=0;i<n;++i){
            if(color[i]==0) {
                color[i]=1;
                q.push(i);
                while(!q.empty()){
                    cur=q.front();
                    q.pop();
                    for(auto next:graph[cur]){
                        if(color[next]==0){
                            color[next]=color[cur]==1?2:1;
                            q.push(next);
                        }
                        else if(color[next]==color[cur]) return false;
                    }
                }
            }   
        }
        return true;
    }
};
```
### 210. Course Schedule II
```cpp

```
## 复杂数据结构

### UF - union-find 
> 684. Redundant Connection
```cpp
class Solution {
public:
    vector<int> findRedundantConnection(vector<vector<int>>& edges) {
        int n=edges.size();
        vector<int> par(n);
        for(int i=0;i<n;++i) par[i]=i;
        int tree1, tree2;
        for(auto edge:edges){
            tree1=uf_find(edge[0]-1, par);
            tree2=uf_find(edge[1]-1, par);
            if(tree1==tree2) return edge;
            uf_union(tree1, tree2, par);
        }
        return vector<int>{};   
    }
    int uf_find(int ch, vector<int>& par){
        if(par[ch]==ch) return ch;
        return par[ch]=uf_find(par[ch], par);
    }
    void uf_union(int tree1, int tree2, vector<int>& par){
        par[tree1]=tree2;
    }
}
```
### LRU 
146. LRU Cache

