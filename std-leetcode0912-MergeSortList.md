### [148\. 排序链表](https://leetcode-cn.com/problems/sort-list/)

Difficulty: **中等**


给你链表的头结点 `head` ，请将其按 **升序** 排列并返回 **排序后的链表** 。

**进阶：**

*   你可以在 `O(n log n)` 时间复杂度和常数级空间复杂度下，对链表进行排序吗？

**示例 1：**

```
输入：head = [4,2,1,3]
输出：[1,2,3,4]
```

**示例 2：**

```
输入：head = [-1,5,3,4,0]
输出：[-1,0,3,4,5]
```

**示例 3：**

```
输入：head = []
输出：[]
```

**提示：**

*   链表中节点的数目在范围 `[0, 5 * 10<sup>4</sup>]` 内
*   `-10<sup>5</sup> <= Node.val <= 10<sup>5</sup>`


#### Solution

Language: ****

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
    ListNode* sortList(ListNode* head) {

        if(head==nullptr || head->next==nullptr) return head;
        ListNode *fast=head->next, *slow=head; // when head->size() equals 2, fast assignned with head
                                               // instead of head->next leads function stack overflow
        while(fast!=nullptr && fast->next!=nullptr){
            fast=fast->next->next;
            slow=slow->next;
        }

        ListNode *rhead=sortList(slow->next);
        slow->next=nullptr;
        head=sortList(head);
        
        return mergeTwoSortedList(head, rhead);
    }
    ListNode* mergeTwoSortedList(ListNode* h1, ListNode* h2) {

        ListNode head, *cur=&head;

        while(h1!=nullptr || h2!=nullptr) {

            if(h1==nullptr) {
                cur->next=h2;
                break;
            }
            if(h2==nullptr) {
                cur->next=h1;
                break;
            }

            if(h1->val <= h2->val) {
                cur->next=h1;
                cur=h1;
                h1=h1->next;
            }
            else {
                cur->next=h2;
                cur=h2;
                h2=h2->next;
            }
        }
        return head.next;
    }

};
```