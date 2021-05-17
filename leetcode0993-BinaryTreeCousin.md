## leetcode0993-BinaryTreeCousin

### 解题思路
BFS解法，队列每次存储同一深度的节点，每取出一个节点，先判断是否是亲兄弟，不是的话根据该次遍历（同一深度）可以判断结果。

### 代码

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    bool isCousins(TreeNode* root, int x, int y) {
        queue<TreeNode*> q;
        q.push(root);
        while(!q.empty()){ // BFS
            int flag=0;
            int sz=q.size(); // 队列存储同一深度的节点
            for(int i=0;i<sz;++i){
                TreeNode* node = q.front(); q.pop();
                if(node->left && node->right){
                    if((node->left->val==x || node->left->val==y) && 
                       (node->right->val==x || node->right->val==y))
                       return false;
                }
                if(node->left) q.push(node->left);
                if(node->right) q.push(node->right);
                if(node->val==x || node->val==y)
                    ++flag;
            }
            if(flag==2) return true;
        }
        return false;
    }
};
```