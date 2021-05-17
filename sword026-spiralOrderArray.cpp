class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        vector<int> res;
        if(matrix.size()==0)
            return res;
        int r1, r2, c1, c2;
        r1=0;
        r2=matrix.size()-1;
        c1=0;
        c2=matrix[0].size()-1;
        while(r1<r2 && c1<c2){
            for(int i=c1;i<c2;++i)
                res.push_back(matrix[r1][i]);
            for(int i=r1;i<r2;++i)
                res.push_back(matrix[i][c2]);
            for(int i=c2;i>c1;--i)
                res.push_back(matrix[r2][i]);
            for(int i=r2;i>r1;--i)
                res.push_back(matrix[i][c1]);
            ++r1;--r2;
            ++c1;--c2;
        }
        if(r1==r2){
            for(int i=c1;i<=c2;++i)
                res.push_back(matrix[r1][i]);
        }
        else if(c1==c2){
            for(int i=r1;i<=r2;++i)
                res.push_back(matrix[i][c1]);
        }
        return res;
    }
    vector<int> spiralOrder2(vector<vector<int>>& matrix) {
        vector <int> res;
        if(matrix.empty()) return res;
        int rl = 0, rh = matrix.size()-1; //记录待打印的矩阵上下边缘
        int cl = 0, ch = matrix[0].size()-1; //记录待打印的矩阵左右边缘
        while(true){
            for(int i=cl; i<=ch; i++) res.push_back(matrix[rl][i]);//从左往右
            if(++rl > rh) break; //若超出边界，退出
            for(int i=rl; i<=rh; i++) res.push_back(matrix[i][ch]);//从上往下
            if(--ch < cl) break;
            for(int i=ch; i>=cl; i--) res.push_back(matrix[rh][i]);//从右往左
            if(--rh < rl) break;
            for(int i=rh; i>=rl; i--) res.push_back(matrix[i][cl]);//从下往上
            if(++cl > ch) break;
        }
        return res;
    }
};