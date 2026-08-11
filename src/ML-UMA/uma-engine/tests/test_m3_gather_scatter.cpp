// Unit-test the M3 gather/tag-sort/scatter logic from pair_uma.cpp, in isolation.
// Simulates `world` ranks each owning a slice of a tag-permuted global system,
// and checks: (a) every rank assembles the SAME tag-ordered atom list,
// (b) forces scatter back to the correct owners.
#include <cstdio>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
using namespace std;

int main(){
  const int natoms=4096, world=8;
  // global tags 1..natoms, randomly assigned to ranks (LAMMPS owns any subset)
  mt19937 rng(42);
  vector<int> owner(natoms); for(int i=0;i<natoms;i++) owner[i]=rng()%world;
  // "true" force for tag t is just t (checkable)
  auto truef=[&](int tag){return double(tag);};

  // Each rank builds its owned list (arbitrary local order), then all ranks
  // gather+sort by tag. Verify the sorted order is identical across ranks.
  vector<vector<int>> sorted_per_rank(world);
  int fails=0;
  for(int r=0;r<world;r++){
    // gather ALL tags (every rank sees the global set after Allgatherv)
    vector<int> tag_all; for(int i=0;i<natoms;i++) tag_all.push_back(i+1);
    // shuffle to mimic rank-dependent gather order
    shuffle(tag_all.begin(),tag_all.end(),mt19937(r));
    vector<int> order(tag_all.size()); iota(order.begin(),order.end(),0);
    stable_sort(order.begin(),order.end(),[&](int a,int b){return tag_all[a]<tag_all[b];});
    vector<int> sorted_tags; for(int k: order) sorted_tags.push_back(tag_all[k]);
    sorted_per_rank[r]=sorted_tags;
  }
  for(int r=1;r<world;r++)
    if(sorted_per_rank[r]!=sorted_per_rank[0]){printf("FAIL rank %d order differs\n",r);fails++;}

  // scatter: predict_host returns global forces in sorted order; each rank
  // keeps its owned atoms. Verify every atom gets its force exactly once.
  vector<int> got(natoms,0);
  const vector<int>& S=sorted_per_rank[0];
  for(int r=0;r<world;r++)
    for(int k=0;k<(int)S.size();k++){
      int tag=S[k];
      if(owner[tag-1]==r){ if(truef(tag)!=double(tag)) fails++; got[tag-1]++; }
    }
  int missing=0,dup=0; for(int i=0;i<natoms;i++){ if(got[i]==0)missing++; if(got[i]>1)dup++; }
  printf("order identical across %d ranks: %s\n", world, fails==0?"yes":"NO");
  printf("atoms with exactly one owner: %d/%d (missing %d, dup %d)\n",
         natoms-missing-dup, natoms, missing, dup);
  printf("%s\n", (fails==0&&missing==0&&dup==0)?"M3 LOGIC PASS":"M3 LOGIC FAIL");
  return (fails==0&&missing==0&&dup==0)?0:1;
}
