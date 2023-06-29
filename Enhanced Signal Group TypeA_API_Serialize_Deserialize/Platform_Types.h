/*	- Faculty of Engineering, Ain Shams University.
 *  - Graduation Project Sponsor by Siemens.
 *  - Dr. Mohamed Taher
 *  - Dr. Ahmed Moro
 * 	- Eng. Hossam
 * 	- Eng. Khaled Ahmed
 * 	- Eng. Ansary
 * 	- Eng. Mohamed Gama Saleh Mohamed Gad
 */

#ifndef PLATFORM_TYPES_H_
#define PLATFORM_TYPES_H_

#include <stdbool.h>
#include <stdint.h>


#ifndef _Bool
#define _Bool unsigned char
#endif

#define CPU_TYPE        CPU_TYPE_32
#define CPU_BIT_ORDED   MSB_FIRST
#define CPU_BYTE_ORDED  HIGH_BYTE_FIRST

#ifndef FALSE
#define FALSE  (boolean)false
#endif

#ifndef TRUE
#define TRUE   (boolean)true
#endif

typedef _Bool 			    boolean;
typedef signed char 	    sint8;
typedef unsigned char 	    uint8;
typedef char 			    char_t;
typedef signed short 	    sint16;
typedef unsigned short 	    uint16;
typedef int 			    sint32;
typedef unsigned  		    uint32;
typedef signed long long 	sint64;
typedef unsigned long long	uint64;
typedef float 			    float32;
typedef double 			    float64;


typedef volatile signed char 		vint8;
typedef volatile unsigned char 		vuint8;

typedef volatile signed short 		vint16;
typedef volatile unsigned short 	vuint16;

typedef volatile int 				vint32;
typedef volatile unsigned  			vuint32;

typedef volatile signed long 	    vint64;
typedef volatile unsigned long   	vuint64;

#endif