#include <CL/cl.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include "timers.h"
#include <string.h>
#include "opencl_errors.h"

int NDIM = 2048;

float *a;
float *b;
float *c;

int print_matrix = 0;
int validation = 0;
int is_gpu = 0;

const char* kernel_source = "__kernel void matmul(__global const float* A, "
                            "__global const float* B, "
                            "__global float* C, "
                            "int size,"
                            "int start_row) {"
                            "  int i = get_global_id(0);"
                            "  int j, k;"
                            "  int c_offset = (start_row + i) * size;"
                            "  float acc;"
                            "  if( i + start_row >= size ) return;"
                            "  for( j = 0; j < size; j++ ) {"
                            "    acc = 0.0f;"
                            "    for( k = 0; k < size; k++ ) {"
                            "      acc += A[i * size + k] * B[k * size + j];"
                            "    }"
                            "    C[c_offset + j] = acc;"
                            "  }"
                            "}";

/************************** DO NOT TOUCH BELOW HERE ******************************/

void free_arrays() {
	free(a);
	a = NULL;
	free(b);
	b = NULL;
	free(c);
	c = NULL;
}

void check_mat_mul( float* c, float* a, float* b )
{
	int i, j, k;
	float sum;
	int validated = 1;

	printf("Validating the result..\n");
	
	// C = AB
	for( i = 0; i < NDIM; i++ )
	{
		for( j = 0; j < NDIM; j++ )
		{
			sum = 0;
			for( k = 0; k < NDIM; k++ )
			{
				sum += a[i * NDIM + k] * b[k * NDIM + j];
			}

			if( c[i * NDIM + j] != sum )
			{
				printf("c[%d][%d] is differ(value=%lf correct_value=%lf)!!\n", i, j, c[i * NDIM + j], sum );
				validated = 0;
			}
		}
	}

	printf("Validation : ");
	if( validated )
		printf("SUCCESSFUL.\n");
	else
		printf("FAILED.\n");
}

void print_mat( float* mat )
{
	int i, j;

	for( i = 0; i < NDIM; i++ )
	{
		for( j = 0; j < NDIM; j++ )
		{
			printf("%8.2lf ", mat[i * NDIM + j]);
		}
		printf("\n");
	}
}

void print_help(const char* prog_name)
{
	printf("Usage: %s [-pvh]\n", prog_name );
	printf("\n");
	printf("OPTIONS\n");
	printf("  -p : print matrix data.\n");
	printf("  -v : validate matrix multiplication.\n");
	printf("  -h : print this page.\n");
}

void parse_opt(int argc, char** argv)
{
	int opt;

	while( (opt = getopt(argc, argv, "d:pvhikjs:")) != -1 )
	{
		switch(opt)
		{
		case 'd':
			if(strcmp("gpu", optarg) == 0) {
				is_gpu = 1;
			}
			break;

		case 's':
			NDIM = atoi(optarg);
			break;
		case 'p':
			// print matrix data.
			print_matrix = 1;
			break;

		case 'v':
			// validation
			validation = 1;
			break;

		case 'h':
		default:
			print_help(argv[0]);
			exit(0);
			break;
		}
	}
}

void check_error(cl_int error_code, int lineno) {
	if(error_code != CL_SUCCESS) {
		printf("OpenCL error occured in line %d! %s\n", lineno, get_opencl_error_message(error_code));
		free_arrays();
		exit(1);
	}
}

int main(int argc, char** argv)
{
	int i, j;
	long k = 1L;

	parse_opt( argc, argv );

	size_t mat_size = NDIM * NDIM;
	size_t mem_size = sizeof(float) * mat_size;

	a = (float *)malloc(mem_size);
	b = (float *)malloc(mem_size);
	c = (float *)malloc(mem_size);

	for( i = 0; i < NDIM; i++ )
	{
		for( j = 0; j < NDIM; j++ )
		{
			a[i * NDIM + j] = k;
			b[i * NDIM + j] = k;
			c[i * NDIM + j] = k * 2;
			k++;
		}
	}

	timer_start(1);

	cl_platform_id platform;
	cl_platform_id* platforms;
	cl_device_id device;
	cl_int error;
	cl_uint platform_count;

	// OpenCL 변수 정의
	error = clGetPlatformIDs(1, &platform, NULL); check_error(error, __LINE__);
	clGetPlatformIDs(1, NULL, &platform_count);
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platform_count); clGetPlatformIDs(platform_count, platforms, NULL);
	error = clGetDeviceIDs(platform, is_gpu? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device, NULL); check_error(error, __LINE__);

	cl_context context = clCreateContext(0, 1, &device, NULL, NULL, &error); check_error(error, __LINE__);
	cl_command_queue command_queue = clCreateCommandQueue(context, device, 0, &error); check_error(error, __LINE__);

	size_t kernel_source_length = strlen(kernel_source);
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_length, &error); check_error(error, __LINE__);

	error = clBuildProgram(program, 1, &device, NULL, NULL, NULL); check_error(error, __LINE__);
	cl_kernel kernel = clCreateKernel(program, "matmul", &error); check_error(error, __LINE__);

	size_t one_line_size = sizeof(float) * NDIM;
	size_t global[1] = { 2048 };
	size_t local[1] = { 16 };
	cl_mem buffer_a = clCreateBuffer(context, CL_MEM_READ_ONLY, one_line_size * global[0], NULL, &error); check_error(error, __LINE__);
	cl_mem buffer_b = clCreateBuffer(context, CL_MEM_READ_ONLY, mem_size, NULL, &error); check_error(error, __LINE__);
	cl_mem buffer_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, mem_size, NULL, &error); check_error(error, __LINE__);
	int idx;
	for( idx = 0; idx < NDIM; idx += global[0] ) {
		int allocate_row_size = (idx + global[0]) > NDIM ? NDIM - idx : global[0];
		// enqueue buffer
		error = clEnqueueWriteBuffer(command_queue, buffer_a, CL_FALSE, 0, one_line_size * allocate_row_size, (void*)(a + (NDIM * idx)), 0, NULL, NULL); check_error(error, __LINE__);
		error = clEnqueueWriteBuffer(command_queue, buffer_b, CL_FALSE, 0, mem_size, (void*)b, 0, NULL, NULL); check_error(error, __LINE__);

		// argument 주입
		error = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffer_a); check_error(error, __LINE__);
		error = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buffer_b); check_error(error, __LINE__);
		error = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buffer_c); check_error(error, __LINE__);
		error = clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&NDIM); check_error(error, __LINE__);
		error = clSetKernelArg(kernel, 4, sizeof(cl_int), (void*)&idx); check_error(error, __LINE__);

		// enqueue execute kernel command
		error = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global, local, 0, NULL, NULL); check_error(error, __LINE__);
	}
	error = clEnqueueReadBuffer(command_queue, buffer_c, CL_TRUE, 0, mem_size, c, 0, NULL, NULL); check_error(error, __LINE__);

	timer_stop(1);

	printf("Time elapsed : %lf sec\n", timer_read(1));


	if( validation )
		check_mat_mul( c, a, b );

	if( print_matrix )
	{
		printf("MATRIX A: \n");
		print_mat(a);

		printf("MATRIX B: \n");
		print_mat(b);

		printf("MATRIX C: \n");
		print_mat(c);
	}

	free_arrays();
	return 0;
}
