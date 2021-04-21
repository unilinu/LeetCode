int dp(string& s, int k);
bool maybe(char a, char b);
class Solution {
public:
    int numDecodings(string s) {
        //未考虑解码方式为0的情况

        return dp(s, s.size());
    }
};
int dp(string& s, int k){
    if(k==0 || k==1) return 1;
    if(s[k-1]=='0')
        return dp(s,k-2);
    else if(s[k-2]=='0')
        return dp(s, k-3);
    else{

        int res=dp(s, k-2);
        if(maybe(s[k-2], s[k-1]))
            res+=res;
        if(k==2) return res;
        if(maybe(s[k-3], s[k-2]))
            res+=dp(s, k-3);
        return res;
    }
}
bool maybe(char a, char b){
    int ia=a-'0';
    int ib=b-'0';
    int comb = ia*10+ib;
    return comb>=11 && comb<=26;
}