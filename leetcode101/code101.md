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
        vector<int> candy(ratings.size(), 1);
        for(int i=1; i<ratings.size(); ++i){
            if(ratings[i]>ratings[i-1])
                candy[i]=candy[i-1]+1;
        }
        for(int i=ratings.size()-2; i>=0; --i){
            if(ratings[i]>ratings[i+1] && candy[i]<=candy[i+1])
                candy[i]=candy[i+1]+1;
        }
        for(int i=1;i<candy.size();++i)
            candy[0]+=candy[i];
        return candy[0];
    }
};
```



