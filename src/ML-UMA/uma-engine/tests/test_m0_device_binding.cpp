#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
static int screen=1, comm_me_g=0;
static int pick(int comm_me,int ndev){
  int local_rank=comm_me;
  for(const char*v:{"SLURM_LOCALID","OMPI_COMM_WORLD_LOCAL_RANK",
                    "MV2_COMM_WORLD_LOCAL_RANK","LOCAL_RANK"}){
    const char*lr=std::getenv(v);
    if(lr==nullptr||*lr=='\0') continue;
    char*end=nullptr; const long parsed=std::strtol(lr,&end,10);
    if(end==lr||*end!='\0'||parsed<0) continue;
    local_rank=static_cast<int>(parsed); break;
  }
  return (ndev>0&&local_rank>=0)?(local_rank%ndev):0;
}
static void clear(){for(const char*v:{"SLURM_LOCALID","OMPI_COMM_WORLD_LOCAL_RANK","MV2_COMM_WORLD_LOCAL_RANK","LOCAL_RANK"})unsetenv(v);}
static int fails=0;
static void chk(const char*n,int got,int want){ if(got!=want){printf("  FAIL %-42s got %d want %d\n",n,got,want);fails++;} else printf("  ok   %-42s -> %d\n",n,got);}
int main(){
  clear(); chk("no env, me=3, ndev=1 (launcher pins)",pick(3,1),0);
  clear(); chk("no env, me=3, ndev=4",pick(3,4),3);
  clear(); chk("no env, me=5, ndev=4 (wrap)",pick(5,4),1);
  clear(); setenv("SLURM_LOCALID","2",1); chk("LOCALID=2, me=7, ndev=4",pick(7,4),2);
  clear(); setenv("SLURM_LOCALID","",1); chk("LOCALID empty -> fall back to me",pick(3,4),3);
  clear(); setenv("SLURM_LOCALID","-1",1); chk("LOCALID=-1 rejected -> me",pick(3,4),3);
  clear(); setenv("SLURM_LOCALID","abc",1); chk("LOCALID=abc rejected -> me",pick(3,4),3);
  clear(); setenv("SLURM_LOCALID","2x",1); chk("LOCALID=2x trailing junk -> me",pick(3,4),3);
  clear(); setenv("LOCAL_RANK","1",1); chk("LOCAL_RANK=1, me=6, ndev=4",pick(6,4),1);
  clear(); setenv("SLURM_LOCALID","abc",1); setenv("LOCAL_RANK","2",1);
           chk("bad LOCALID falls through to LOCAL_RANK",pick(7,4),2);
  clear(); chk("ndev=0 guard (no div-by-zero)",pick(3,0),0);
  clear(); setenv("SLURM_LOCALID","0",1); chk("LOCALID=0 is valid, not 'empty'",pick(3,4),0);
  printf("\n%s (%d failures)\n", fails? "TESTS FAILED":"ALL BINDING TESTS PASS", fails);
  return fails?1:0;
}
