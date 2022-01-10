### [692\. 前K个高频单词](https://leetcode-cn.com/problems/top-k-frequent-words/)

Difficulty: **中等**


给一非空的单词列表，返回前 _k _个出现次数最多的单词。

返回的答案应该按单词出现频率由高到低排序。如果不同的单词有相同出现频率，按字母顺序排序。

**示例 1：**

```
输入: ["i", "love", "leetcode", "i", "love", "coding"], k = 2
输出: ["i", "love"]
解析: "i" 和 "love" 为出现次数最多的两个单词，均为2次。
    注意，按字母顺序 "i" 在 "love" 之前。
```

**示例 2：**

```
输入: ["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"], k = 4
输出: ["the", "is", "sunny", "day"]
解析: "the", "is", "sunny" 和 "day" 是出现次数最多的四个单词，
    出现次数依次为 4, 3, 2 和 1 次。
```

**注意：**

1.  假定 _k_ 总为有效值， 1 ≤ _k_ ≤ 集合元素数。
2.  输入的单词均由小写字母组成。

**扩展练习：**

1.  尝试以 _O_(_n_ log _k_) 时间复杂度和 _O_(_n_) 空间复杂度解决。


#### Solution

Language: C++

```cpp
class Solution {
public:
    vector<string> topKFrequent(vector<string>& words, int k) {
        // ordered_map or priority_queue
    }
    vector<string> topKFrequent(vector<string>& words, int k) {
        // coding my partition 
        // std partition twice
    }
    vector<string> topKFrequent(vector<string>& words, int k) {
        unordered_map<string, int> map;
        for(string& word:words)
            ++map[word];

        vector<string> keys;
        vector<int> values;
        vector<int> index;
        int tmp=-1;
        for(auto& pr:map){
            keys.push_back(pr.first);
            index.push_back(++tmp);
            values.push_back(pr.second);
//            cout<<pr.first<<' '<<pr.second<<' '<<tmp<<endl;
        }
        auto first=index.begin(), last=index.end();
        while(first != last){ // quickswap
            // int pivot_index=*(first+(last-first)/2);
            int pivot_index=*(first + (last - first) / 2); // pivot need valid
            auto cmp = [&keys, &values, pivot_index](int em_index) -> bool {
                if(values[pivot_index] < values[em_index])
                    return true;
                else if(values[pivot_index] == values[em_index] &&
                        keys[pivot_index].compare(keys[em_index])>0)
                    return true;
                return false;
            };
            auto middle_index=partition(first, last, cmp);
            if(middle_index-index.begin()==k)
                break;
            else if(middle_index-index.begin()>k){
                last=middle_index;
            }
            else {
                iter_swap(middle_index, find(middle_index, last, pivot_index));
                first= middle_index + 1;
            }
            // cout<<*first<<' '<<*(last-1)<<endl;

        }
        auto cmp=[&keys, &values](int i, int j) -> bool {
            if(values[j] < values[i])
                return true;
            else if(values[i] == values[j] && (keys[i].compare(keys[j])<0))
                return true;
            return false;
        };
        sort(index.begin(), index.begin()+k, cmp);
        vector<string> res;
        for(int i=0;i<k;++i)
            res.push_back(keys[index[i]]);
        return res;
    }
};

```