#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

// Break A array into parts, work on one part in one thread
// when all threads are done working on their part of A,
// calculate offset to be added by adding up precomputed
// local_prefix entries of all previous threads.
// Works because static scheduling and no. of parts = no. of threads
void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan
  if(n == 0) return;
  
  int tid, nthreads, part_length;
  prefix_sum[0] = 0;
  long* local_prefix;

  #pragma omp parallel private(tid)
  {

    // all threads wait at the end of single block
    #pragma omp single
    {
      nthreads = omp_get_num_threads();
      printf("Number of threads = %d\n", nthreads);
      part_length = ceil(n / (double) nthreads);
      printf("Part length = %d\n", part_length);
      local_prefix = new long[nthreads+1];
      local_prefix[0] = 0;
    }

    tid = omp_get_thread_num();

    // operate on parts of A
    #pragma omp for schedule(static, 1) nowait
    for(int t = 0; t < nthreads; t++) {
      long base = t*part_length;
      long limit = MIN(n, base + part_length);
      for(long i = base+1; i < limit; i++) {
        prefix_sum[i] = prefix_sum[i-1] + A[i-1];
      }
      local_prefix[t+1] = prefix_sum[limit-1] + A[limit-1];
    }

    #pragma omp barrier

    // update local copy of offset to be added
    long offset = 0;
    for(int t = 0; t <= tid; t++)
      offset += local_prefix[t];

    // add offset to get final value
    #pragma omp for schedule(static, 1) nowait
    for(int t = 0; t < nthreads; t++) {
      long base = t*part_length;
      long limit = MIN(n, base + part_length);
      for(long i = base; i < limit; i++)
        prefix_sum[i] += offset;
    }


    #pragma omp single
    delete [] local_prefix;
    
  }

}

void print_result(long* b, long n) {
  for(int i = 0; i < n; i++)
    printf("%ld\n", b[i]);
  printf("\n");
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime(), st, pt;
  scan_seq(B0, A, N);
  st = omp_get_wtime() - tt;
  printf("sequential-scan = %fs\n", st);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  pt = omp_get_wtime() - tt;
  printf("parallel-scan   = %fs\n", pt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  printf("speedup of parallel scan over sequential scan %lf\n", st/pt);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
