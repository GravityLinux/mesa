/* SPDX-License-Identifier: MIT
 * Opt-in CPU/GPU submission diagnostics. No file writes on event paths.
 */
#include "agx_timeline.h"
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
static struct tl_header *header;
static struct tl_record *records;
static pthread_once_t once = PTHREAD_ONCE_INIT;
static uint64_t ids, next_clock, present_frame;
uint64_t agx_timeline_frame(void) { return __atomic_load_n(&present_frame,__ATOMIC_RELAXED); }
void agx_timeline_presented(void) { __atomic_add_fetch(&present_frame,1,__ATOMIC_RELAXED); }
static _Thread_local uint64_t api_id, driver_id;
uint64_t agx_timeline_driver(void) { return driver_id; }
uint64_t agx_timeline_set_driver(uint64_t id) { uint64_t old=driver_id;driver_id=id;return old; }
static _Thread_local uint32_t thread_id;
static uint64_t mono(void) {
 struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
 return (uint64_t)t.tv_sec*1000000000ull+t.tv_nsec;
}
static uint64_t raw_counter(void) {
#if defined(__aarch64__)
 uint64_t ticks; __asm__ volatile("isb; mrs %0, cntvct_el0" : "=r"(ticks)); return ticks;
#else
 return mono();
#endif
}
static uint64_t raw_frequency(void) {
#if defined(__aarch64__)
 uint64_t hz; __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(hz)); return hz;
#else
 return 1000000000;
#endif
}
static void initialize(void) {
 const char *path=getenv("HK_TIMELINE_FILE"); if (!path || !*path) return;
 char filename[4096]; snprintf(filename,sizeof(filename),"%s.%u",path,(unsigned)getpid());
 int fd=open(filename,O_CREAT|O_EXCL|O_RDWR|O_CLOEXEC|O_NOFOLLOW,0600);
 if (fd<0) { perror("timeline create"); return; }
 unsigned long capacity=TL_DEFAULT_CAPACITY;
 const char *setting=getenv("HK_TIMELINE_CAPACITY");
 if(setting) { unsigned long value=strtoul(setting,NULL,0); if(value>=4096 && value<=(1u<<24)) capacity=value; }
 size_t bytes=TL_HEADER_SIZE+(size_t)capacity*TL_RECORD_SIZE;
 if (ftruncate(fd,bytes)) { close(fd); return; }
 void *p=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);close(fd);
 if (p==MAP_FAILED) return;
 /* Fault in the bounded recording storage during startup, outside windows. */
 memset(p,0,bytes); header=p;records=(void *)((char *)p+TL_HEADER_SIZE);
 header->magic=TL_MAGIC;header->version=TL_VERSION;header->header_size=TL_HEADER_SIZE;
 header->record_size=TL_RECORD_SIZE;header->capacity=capacity;
 header->pid=getpid();
header->clock_domain=TL_MONOTONIC_NS;header->frequency=1000000000;
 /* Recording remains disabled until the controller grants the capture gate. */
}
void agx_timeline_init(void) { pthread_once(&once,initialize); }
bool agx_timeline_enabled(void) {
 agx_timeline_init();return header && __atomic_load_n(&header->enabled,__ATOMIC_ACQUIRE);
}
uint64_t agx_timeline_id(void) { return __atomic_add_fetch(&ids,1,__ATOMIC_RELAXED); }
static void emit_at(unsigned type,unsigned flags,uint64_t id,uint64_t stamp,
 uint64_t a,uint64_t b,uint64_t c,uint64_t d,uint64_t e,uint64_t f,uint64_t g) {
 uint64_t i=__atomic_fetch_add(&header->written,1,__ATOMIC_RELAXED);
 if (i>=header->capacity) {__atomic_fetch_add(&header->dropped,1,__ATOMIC_RELAXED);return;}
 if (!thread_id) thread_id=syscall(SYS_gettid);
 struct tl_record *r=&records[i];
 r->stamp=stamp;r->id=id;r->a=a;r->b=b;r->c=c;r->d=d;r->e=e;r->f=f;r->g=g;
 r->pid=header->pid;r->tid=thread_id;r->type=type;r->flags=flags;
 __atomic_store_n(&r->seq,i+1,__ATOMIC_RELEASE);
}
uint64_t agx_timeline_emit(unsigned type,unsigned flags,uint64_t id,
 uint64_t a,uint64_t b,uint64_t c,uint64_t d,uint64_t e,uint64_t f,uint64_t g) {
 if (!agx_timeline_enabled()) return 0;
 uint64_t stamp=mono(),limit=__atomic_load_n(&next_clock,__ATOMIC_RELAXED);
 if (stamp>=limit && __atomic_compare_exchange_n(&next_clock,&limit,
      stamp+header->frequency,false,__ATOMIC_RELAXED,__ATOMIC_RELAXED)) {
  uint64_t before=mono(),ticks=raw_counter(),after=mono();
  emit_at(TL_CLOCK,0,0,after,before,after,raw_frequency(),ticks,0,0,0);
 }
 emit_at(type,flags,id,stamp,a,b,c,d,e,f,g);return id;
}
uint64_t agx_timeline_hash(const void *data,size_t size) {
 const unsigned char *p=data;uint64_t h=14695981039346656037ull;
 for(size_t i=0;i<size;i++) h=(h^p[i])*1099511628211ull;
 return h;
}
uint64_t agx_timeline_api(void) {return api_id;}
struct agx_timeline_scope agx_timeline_scope_begin(unsigned op,uint64_t object,
 uint64_t count,uint64_t fence,uint64_t timeout,uint64_t flags) {
 struct agx_timeline_scope s={0}; if(!agx_timeline_enabled())return s;
 s=(struct agx_timeline_scope){agx_timeline_id(),api_id,op};api_id=s.id;
 agx_timeline_emit(TL_API_BEGIN,op,s.id,s.parent,object,count,fence,timeout,flags,agx_timeline_frame());
 return s;
}
void agx_timeline_scope_end(struct agx_timeline_scope *s,int result) {
 if(!s->id)return;
 agx_timeline_emit(TL_API_END,s->op,s->id,s->parent,(uint64_t)(int64_t)result,0,0,0,0,0);
 api_id=s->parent;
}

void agx_timeline_error(unsigned error, uint64_t detail) {
 agx_timeline_init();if(header)__atomic_add_fetch(&header->errors,1,__ATOMIC_RELAXED);
 agx_timeline_emit(TL_ERROR,error,0,detail,0,0,0,0,0,0);
}
