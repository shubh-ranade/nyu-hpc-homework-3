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

void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan
  if(n == 0) return;
  
  int tid, nthreads, part_length;
  long s = 0;
  prefix_sum[0] = 0;
  long* local_prefix;

  // for(long tid = 1; tid <= (long) num_threads; tid++){
  //   long start = fmin(n, tid * thread_segment + 1);
  //   long end = fmin(n, (((tid+1) * thread_segment) + 1));

  //   for (long i = start; i < end; i++){
  //     //printf("%ld\n", i);
  //     //if(i ==9) printf("%ld\n", prefix_sum[i]);
  //     prefix_sum[i] = prefix_sum[i] +  prefix_sum[start-1];
  //     //if(i==9) printf("%ld\n", prefix_sum[i]);
  //   }
  // }

  #pragma omp parallel private(tid)
  {
    // if(tid == 0) {
    //   nthreads = omp_get_num_threads();
    //   printf("Number of threads = %d\n", nthreads);
    //   part_length = n % nthreads ? (n / nthreads) + 1 : n / nthreads;
    //   printf("Part length = %d\n", part_length);
    //   local_prefix = new long[nthreads+1];
    //   local_prefix[0] = 0;
    // }

    #pragma omp single
    {
      nthreads = omp_get_num_threads();
      printf("Number of threads = %d\n", nthreads);
      part_length = n % nthreads ? (n / nthreads) + 1 : n / nthreads;
      printf("Part length = %d\n", part_length);
      local_prefix = new long[nthreads+1];
      local_prefix[0] = 0;
    }

    tid = omp_get_thread_num();


    // #pragma omp barrier

    // long* local_prefix = new long[part_length];
    // local_prefix[0] = 0;

    // #pragma omp for schedule(static)
    // for(int t = 0; t < nthreads; t++) {
    //   long base = t*part_length;
    //   long limit = std::min(n, base + part_length);
    //   local_prefix[0] = base > 0 ? A[base-1] : 0;
    //   for(long i = base+1, j = 1; i < limit; i++, j++) {
    //     printf("1 tid = %d, local_prefix[i=%ld, j=%ld] = %ld, %ld\n", tid, i, j, local_prefix[j], local_prefix[0]);
    //     local_prefix[j] = local_prefix[j-1] + A[i-1];
    //     prefix_sum[i] = local_prefix[j];
    //     // printf("tid = %d, t = %d, i = %d, j = %d\n", tid, t, i, j);
    //   }
    // }
    // printf("done\n");

    // #pragma omp for schedule(static, part_length)
    // for(long i = 0; i < n; i++) {
    //   long j = i % part_length;
    //   if(j > 0) {
    //     local_prefix[j] = local_prefix[j-1] + A[i-1];
    //   } else if(j == 0 && i > 0) {
    //     local_prefix[j] = A[i-1];
    //   }
    // }

    // #pragma omp for schedule(static, part_length) ordered
    // for(long i = 0; i < n; i++) {
    //   long j = i % part_length;
    //   long start_offset = (i/part_length) * part_length - 1;
    //   #pragma omp ordered
    //   {
    //     if(tid > 0)
    //       prefix_sum[i] = local_prefix[j] + prefix_sum[start_offset];
    //     else
    //       prefix_sum[i] = local_prefix[j];
    //   }
    // }

    long local_sum = 0;
    #pragma omp for schedule(static, 1) nowait
    for(int t = 0; t < nthreads; t++) {
      long base = t*part_length;
      long limit = MIN(n, base + part_length);
      for(long i = base+1; i < limit; i++) {
        // local_sum += A[i-1];
        // prefix_sum[i] = local_sum;
        prefix_sum[i] = prefix_sum[i-1] + A[i-1];
      }
      // local_prefix[t+1] = local_sum + A[limit-1];
      local_prefix[t+1] = prefix_sum[limit-1] + A[limit-1];
      // printf("tid %d, localpref %ld\n", tid, local_prefix[t+1]);
    }

    #pragma omp barrier

    long offset = 0;
    for(int t = 0; t <= tid; t++)
      offset += local_prefix[t];

    // #pragma omp barrier
    // // printf("tid %d, offset %ld\n", tid, offset);
    // #pragma omp single
    // for(long i = 0; i < n; i++)
    //   // printf("prefix sum [%ld] %ld\n", i, prefix_sum[i]);
    // #pragma omp barrier

    #pragma omp for schedule(static)
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
  // for (long i = 0; i < N; i++) A[i] = rand();
  for (long i = 0; i < N; i++) A[i] = i + 1;
  // print_result(A, N);

  double tt = omp_get_wtime(), st, pt;
  scan_seq(B0, A, N);
  st = omp_get_wtime() - tt;
  printf("sequential-scan = %fs\n", st);
  // print_result(B0, N);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  pt = omp_get_wtime() - tt;
  printf("parallel-scan   = %fs\n", pt);
  // print_result(B1, N);
  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  printf("speedup of parallel scan over sequential scan %lf\n", st/pt);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
