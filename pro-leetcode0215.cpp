// 数组中的第K个最大元素
// 在未排序的数组中找到第 k 个最大的元素。请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。

class Solution {
public:
    int findKthLargest1(vector<int>& nums, int k) { // 最小堆
        auto cmp = [](int i, int j) { return (i>j); };
        priority_queue<int, vector<int>, decltype(cmp) > pq(nums.begin(), nums.begin()+k, cmp);
        for(int i=k;i<nums.size();++i){
            pq.push(nums[i]);
            pq.pop();
        }
        return pq.top();
    }
    int findKthLargest(vector<int>& nums, int k) { // 快速交换 
        auto first=nums.begin(), last=nums.end();
        while(true){
            auto pivot = *std::next(first, std::distance(first,last)/2); // auto pivot=*first;
            auto middle1=partition(first, last, [pivot](const int& em){ return em>pivot; }); // 主要需要交换两次，避免死循环：pivot为max/min/equal
            auto middle2=partition(middle1, last, [pivot](const int& em){ return em>=pivot; });
    //        cout<<pivot<<' '<<*middle<<endl;
            if(distance(first, first+k) <= distance(first, middle1))
                last=middle1;
            else if(distance(first, first+k) <= distance(first, middle2)){
                return *middle1;

            }
            else {
                k-=(int)distance(first, middle2);
                first=middle2;
            }
        }
    }
};