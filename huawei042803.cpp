#include <iostream>
#include <vector>
#include <unordered_map>

using namespace std;

int main(){
    int N,M,x;
    cin>>M>>N>>x;
    unordered_map<string, vector<int>> demand;
    unordered_map<string, vector<int>> supply;
    for(int i=0;i<M;++i){
        string power;
        getline(cin,power);
        int start=0;
        for(int j=0;j<power.size();++j) {
            if(power[j]==' '){
                supply[power.substr(start, j)].push_back(i);
                start=j+1;
            }
        }

    }
    for(auto key : supply){
        cout<<key.first<<endl<<endl;
        for(int index:key.second)
            cout<<index<<' ';
    }




    return 0;
}