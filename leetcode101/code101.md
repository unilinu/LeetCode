## 贪心

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

## 双指针

### 左右指针

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





### 88. Merge Sorted Array

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
## 二分查找

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

## 排序

### [215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)

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


## 搜索

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



### BFS

#### 934. Shortest Bridge

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

## DP

### 二维

#### 542. 01 Matrix

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
## 分治
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



## 巧解数学

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

#### [233. 数字 1 的个数](https://leetcode-cn.com/problems/number-of-digit-one/)

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

## 妙用数据结构

### 769. Max Chunks To Make Sorted
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

### 155. Min Stack
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
### 739. Daily Temperatures
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

### 23. Merge k Sorted Lists

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
### 218. 天际线问题
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
