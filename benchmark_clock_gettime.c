#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "computepi.h"

#define CLOCK_ID CLOCK_MONOTONIC_RAW
#define ONE_SEC 1000000000.0
#define PI acos(1.0);

double get_avgofCI(double* datas , double loop_size)
{
    double ci_max = 0.0;
    double ci_min = 0.0;
    double sd = 0.0;
    double avg = 0.0;
    double output = 0.0;
    for(int i=0; i < loop_size ; i++)
    {
        avg += datas[i];
    }
    avg = avg / loop_size;
    
    for(int i=0;i<loop_size ; i++)
    {
        const double tmp = datas[i] - avg; 
        sd += (tmp * tmp);
    }
    sd = sqrt(sd / loop_size);

    ci_max = avg + (1.96 * sd );
    ci_min = avg - (1.96 * sd ); 

    int valid=0;
    for(int i = 0 ; i < loop_size ; i++)
    {
        if(datas[i] >= ci_min && datas[i] <= ci_max)
        {
            valid++;
            output += datas[i];
        }
    }

    output = output / (double)valid;

    return output;
}


int main(int argc, char const *argv[])
{
    struct timespec start = {0, 0};
    struct timespec end = {0, 0};

    if (argc < 2) return -1;

    int N = atoi(argv[1]);
    int i, loop = 25;
    double datas[25];
    // Baseline
    for(i = 0; i < loop; i++) {
        clock_gettime(CLOCK_ID, &start);
        compute_pi_baseline(N);
        clock_gettime(CLOCK_ID, &end);
        datas[i] = (double) (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)/ONE_SEC;
    }
#if defined(USE_CI)
    {
        double result = get_avgofCI(datas, loop);
        printf("%lf " , result);
    }
#else
    {
        double avg = 0.0;
        for(int i=0;i<loop;i++)
        {
            avg += datas[i];
        }
        avg = avg / (double)loop;
        printf("%lf " , avg);
    }

#endif

    // OpenMP with 2 threads
    for(i = 0; i < loop; i++) {
        clock_gettime(CLOCK_ID, &start);
        compute_pi_openmp(N , 2);
        clock_gettime(CLOCK_ID, &end);
        datas[i] = (double) (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)/ONE_SEC;
    }
#if defined(USE_CI)
    {
        double result = get_avgofCI(datas, loop);
        printf("%lf " , result);
    }
#else
    {
        double avg = 0.0;
        for(int i=0;i<loop;i++)
        {
            avg += datas[i];
        }
        avg = avg / (double)loop;
        printf("%lf " , avg);
    }

#endif

    // OpenMP with 4 threads
    for(i = 0; i < loop; i++) {
        clock_gettime(CLOCK_ID, &start);
        compute_pi_openmp(N , 4);
        clock_gettime(CLOCK_ID, &end);
        datas[i] = (double) (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)/ONE_SEC;
    }
#if defined(USE_CI)
    {
        double result = get_avgofCI(datas, loop);
        printf("%lf " , result);
    }
#else
    {
        double avg = 0.0;
        for(int i=0;i<loop;i++)
        {
            avg += datas[i];
        }
        avg = avg / (double)loop;
        printf("%lf " , avg);
    }

#endif
    // AVX SIMD
    for(i = 0; i < loop; i++) {
        clock_gettime(CLOCK_ID, &start);
        compute_pi_avx(N);
        clock_gettime(CLOCK_ID, &end);
        datas[i] = (double) (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)/ONE_SEC;
    }
#if defined(USE_CI)
    {
        double result = get_avgofCI(datas, loop);
        printf("%lf " , result);
    }
#else
    {
        double avg = 0.0;
        for(int i=0;i<loop;i++)
        {
            avg += datas[i];
        }
        avg = avg / (double)loop;
        printf("%lf " , avg);
    }

#endif
    // AVX SIMD + Loop unrolling
    for(i = 0; i < loop; i++) {
        clock_gettime(CLOCK_ID, &start);
        compute_pi_avx_unroll(N);
        clock_gettime(CLOCK_ID, &end);
        datas[i] = (double) (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)/ONE_SEC;
    }
#if defined(USE_CI)
    {
        double result = get_avgofCI(datas, loop);
        printf("%lf\n" , result);
    }
#else
    {
        double avg = 0.0;
        for(int i=0;i<loop;i++)
        {
            avg += datas[i];
        }
        avg = avg / (double)loop;
        printf("%lf\n" , avg);
    }
#endif    

    return 0;
}
