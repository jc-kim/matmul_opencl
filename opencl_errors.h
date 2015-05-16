# ifndef __OPENCL_ERRORS_H__
# define __OPENCL_ERRORS_H__
#include <CL/cl.h>

const char* get_opencl_error_message(cl_int code);
# endif
