### 贪心

#### 455 Assign cookies (easy)

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


#### 135 Candy (hard)

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

#### 	605. Can Place Flowers （easy)

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

#### [435. 无重叠区间](https://leetcode-cn.com/problems/non-overlapping-intervals/)

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

#### 452. Arrows to Burst Balloons

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

#### 665. Non-decreasing Array

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

### 双指针

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

