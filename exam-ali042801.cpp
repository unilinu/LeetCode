#include<vector>
#include <iostream>

struct position {
    int row, col;
};

inline position next_position(position cur, char dir){
    if(dir=='D')
        return position{cur.row+1, cur.col};
    else if(dir=='U')
        return position{cur.row-1, cur.col};
    else if(dir=='R')
        return position{cur.row, cur.col+1};
    else
        return position{cur.row, cur.col-1};
}
bool is_2x2area(position cur){ // forward
    // top_left
    if()
}


int main() {
    int n,m;
    cin>>n>>m;
    vector< vector<char> > map(n+5, vector<char>(m+5, 'W'));
    for (int i=1;i<=n;++i){
        map[i][0]='0';
        map[i][m+1]='0';
    }
    for (int j=1;j<=m;++j){
        map[0][j]='0';
        map[n+1][j]='0';
    }
    char dir='R';
    position cur = position {1,1};
    char left=false;
    while(true){
        position next = next_position(cur, dir);
        if ()

    }

    return 0;
}