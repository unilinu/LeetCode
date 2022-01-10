#include <iostream>
#include <vector>
#include <unordered_map>

using namespace std;

int find(vector<int>& par, int val) {
    return val==par[val]?val:par[val]=find(par, par[val]);
}
int main(){
    int n,m;
    cin>>n>>m;
    vector<double> pools(n+1);
    unordered_map<int, vector<int> > map;
    vector<int> par(n+1);
    for (int i = 1; i <= n; ++i) {
        cin>>pools[i];
        par[i]=i;
    }
    for (int i = 0; i < m; ++i) {
        int u,v;
        cin>>u>>v;
        par[find(par, u)]=find(par, v);
    }
    for (int i = 1; i <= n; ++i) {
        int root=find(par, i);
        map[root].push_back(i);
    }
    for(const auto& pr : map) {
//        cout<<pr.first<<": [";
        double sum=0.0;
        for (auto i : pr.second) {
//            cout<<i<<' ';
            sum+=pools[i];
        }
        if(pr.second.empty()) continue;
        double ave = (sum/pr.second.size())+0.005;
        for (auto i : pr.second) {
            pools[i]= ave;
        }
//        cout<<"= "<<ave<<"*"<<pr.second.size()<<"]"<<endl;
    }
    for (int i = 1; i < n; ++i) {
        printf("%.2f ", pools[i]);
    }
    printf("%.2f", pools[n]);

    return 0;
}