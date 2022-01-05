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





