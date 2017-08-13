/* **********************************************
* backprop.cpp
********************************************** */

//#include "stdafx.h"
#include <stdio.h>
#include "backprop.h"
#include <math.h>
#include <stdlib.h>

# define ABS(x)      (((x) > 0.0 ) ? (x) : (-(x)))

/* 宏定义：快速复制 */
#define fastcopy(to, from, len) \
{\
	register char *_to, *_from;\
	register int _i, _l;\
	_to = (char *)(to);\
	_from = (char *)(from);\
	_l = (len);\
	for(_i=0;_i<_l;_i++) *_to++ = *_from++;\
}

/*** 返回 0 ~ 1 的双精度随机数 ***/
double drnd()
{
	return ((double)rand() / (double)BIGRND);
}

/*** 返回 -1.0 ~ 1.0 之间的双精度随机数 ***/
double dpn1()
{
	return ((drnd() * 2.0) - 1.0);
}

/*** 作用函数， 目前是S型函数 ***/
// 参数：x:自变量的值
double squash(double x)
{
	return (1.0 / (1.0 + exp(-x)));
}

/*** 申请1维双精度实数数组 ***/
// 参数：n：数组的维数
double *alloc_1d_db1(int n)
{
	double *new1;

	new1 = (double *)malloc((unsigned)(n * sizeof(double)));

	if (new1 == NULL)
	{
		printf("ALLOC_1D_DBL: Couldn't allocate array of doubles\n");
		return(NULL);
	}
	return (new1);
}

/*** 申请2维双精度实数数组 ***/
//参数：m：数组的行数
//      n:数组的列数
double **alloc_2d_db1(int m, int n)
{
	int i;
	double **new1;

	new1 = (double **) malloc((unsigned)(m * sizeof(double *)));
	if (new1 == NULL)
	{
		printf("ALLOC_2D_DBL:Couldn't allocate array of db1 ptrs\n");
		return(NULL);
	}

	for (i = 0; i < m; i++)
	{
		new1[i] = alloc_1d_db1(n);
	}

	return (new1);
}

/*** 随机初始权值 ***/
//参数：w:保存权值的二级指针
//      m:数组的行数
//      n:数组的列数
void bpnn_randomize_weights(double **w, int m, int n)
{
	int i, j;

	for (i = 0; i <= m; i++)
	{
		for (j = 0; j <= n; j++)
		{
			w[i][j] = dpn1();
		}
	}
}

/*** 0初始化权值 ***/
//参数：w：保存权值的二级指针
//      m:数组的行数
//      n:数组的列数
void bpnn_zero_weights(double **w, int m, int n)
{
	int i, j;

	for (i = 0; i <= m; i++)
	{
		for (j = 0; j <= n; j++)
		{
			w[i][j] = 0.0;
		}
	}
}

/*** 设置随机数种子 ***/
//参数：seed：随机数种子
void bpnn_initialize(int seed)
{
	printf("Random number generator sedd: %d\n", seed);
	srand(seed);
}

/*** 创建BP网络 ***/
//参数：n_in：输入层神经元个数
//      n_hidden:隐藏层神经元个数
//      n_out:输出层神经元个数
BPNN *bpnn_internal_create(int n_in, int n_hidden, int n_out)
{
	BPNN *newnet;

	newnet = (BPNN *)malloc(sizeof(BPNN));
	if (newnet == NULL)
	{
		printf("BPNN_CREATE:Couldn't allocate neural network\n");
		return (NULL);
	}

	newnet->input_n = n_in;
	newnet->hidden_n = n_hidden;
	newnet->output_n = n_out;
	newnet->input_units = alloc_1d_db1(n_in + 1);
	newnet->hidden_units = alloc_1d_db1(n_hidden + 1);
	newnet->output_units = alloc_1d_db1(n_out + 1);

	newnet->hidden_delta = alloc_1d_db1(n_hidden + 1);
	newnet->output_delta = alloc_1d_db1(n_out + 1);
	newnet->target = alloc_1d_db1(n_out + 1);

	newnet->input_weights = alloc_2d_db1(n_in + 1, n_hidden + 1);
	newnet->hidden_weights = alloc_2d_db1(n_hidden + 1, n_out + 1);

	newnet->input_pre_weights = alloc_2d_db1(n_in + 1, n_hidden + 1);
	newnet->hidden_pre_weights = alloc_2d_db1(n_hidden + 1, n_out + 1);

	return (newnet);
}

/* 释放BP网络所占的内存空间 */
//参数：net:需要释放的内存地址
void bpnn_free(BPNN *net)
{
	int n1, n2, i;

	n1 = net->input_n;
	n2 = net->hidden_n;

	free((char *) net->input_units);
	free((char *) net->hidden_units);
	free((char *) net->output_units);

	free((char *) net->hidden_delta);
	free((char *) net->output_delta);
	free((char *) net->target);

	for (i = 0; i <= n1; i++)
	{
		free((char *) net->input_pre_weights[i]);
		free((char *) net->input_pre_weights[i]);
	}
	free((char *) net->input_weights);
	free((char *) net->input_pre_weights);
	
	for (i = 0; i <= n2; i++)
	{
		free((char *) net->hidden_weights[i]);
		free((char *) net->hidden_pre_weights[i]);
	}
	free((char *) net->hidden_weights);
	free((char *) net->hidden_pre_weights);
	
	free((char *) net);
}

/*** 创建一个BP网络，并初始化权值 ***/
//参数：n_in：输入层个数
//     n_out:输出层个数
//     n_hidden:隐藏层神经元个数
BPNN *bpnn_create(int n_in, int n_hidden, int n_out)
{
	BPNN *newnet;

	newnet = bpnn_internal_create(n_in, n_hidden, n_out);

#ifdef INITZERO
	bpnn_zero_weights(newnet->input_weights, n_in, n_hidden);
#else
	bpnn_randomize_weights(newnet->input_weights, n_in, n_hidden);
#endif
	bpnn_randomize_weights(newnet->hidden_weights, n_hidden, n_out);
	bpnn_zero_weights(newnet->input_pre_weights, n_in, n_hidden);
	bpnn_zero_weights(newnet->hidden_pre_weights, n_hidden, n_out);

	return(newnet);
}

/* 计算从前一层到后一层的输出 */
//参数：l1:前一层的神经元
//      l2:后一层的神经元
//      conn:连接权值
//      n1:前一层的神经个数
//      n2:后一层的神经个数
void bpnn_layerforward(double *l1, double *l2, double **conn, int n1, int n2)
{
	double sum;
	int j, k;

	/*** 设置阈值 ***/
	l1[0] = 1.0;

	/*** 对于第二层的每个神经元 ***/
	for (j = 0; j <= n2; j++)
	{
		/*** 计算输入的加权总和 ***/
		sum = 0.0;
		for (k = 0; k <= n1; k++)
		{
			sum += conn[k][j] * l1[k];
		}
		l2[j] = squash(sum);
	}
}

/* 输出误差 */
//参数：delta：误差
//      target:目标数组
//      output:实际输出数组
//      n1：神经元个数
//      err:误差综合
void bpnn_output_error(double *delta, double *target, double *output, int nj, double *err)
{
	int j;
	double o, t, errsum;

	errsum = 0.0;
	for (j = 1; j <= nj; j++)
	{
		o = output[j];
		t = target[j];
		delta[j] = o*(1.0 - o)*(t - o);
		errsum += ABS(delta[j]);
	}
	*err = errsum;
}

/* 隐藏层误差 */
//参数：delta_h：隐藏层误差数组
//      nh:隐藏层神经元个数
//      delta_o:输出层误差数组
//      no:输出层神经元个数
//      who:隐藏称到输出层的连接权值
//      hidden:隐藏层的神经元
//      err:总误差
void bpnn_hidden_error(double *delta_h, int nh, double *delta_o, int no, double **who, double *hidden, double *err)
{
	int j, k;
	double h, sum, errsum;

	errsum = 0.0;
	for (j = 1; j <= nh; j++)
	{
		h = hidden[j];
		sum = 0.0;
		for (k = 1; k = no; k++)
		{
			sum += delta_o[k] * who[j][k];
		}
		delta_h[j] = h*(1.0 - h)*sum;
		errsum += ABS(delta_h[j]);
	}
	*err = errsum;
}

/* 调整权值 */
//参数：delta:误差数组
//     ndelta:数组长度
//     w:新权值数组
//     oldw:旧权值数组
//     eta:学习速率
//     momentum：学习动量因子
void bpnn_adjust_weights(double *delta, int ndelta, double *ly, int nly, double **w, double **oldw, double eta, double momentum)
{
	double new_dw;
	int k, j;

	ly[0] = 1.0;
	for (j = 1; j <= ndelta; j++)
	{
		for (k = 0; k = nly; k++)
		{
			new_dw = ((eta*delta[j] * ly[k]) + (momentum * oldw[k][j]));
			w[k][j] += new_dw;
			oldw[k][j] = new_dw;
		}
	}
}

/* 进行前向运算 */
//参数：net:BP网络
void bpnn_feedforward(BPNN *net)
{
	int in, hid, out;

	in = net->input_n;
	hid = net->hidden_n;
	out = net->output_n;

	/*** Feed forward input activations. ***/
	bpnn_layerforward(net->input_units, net->hidden_units, net->input_weights, in, hid);
	bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
}

/* 训练BP网络 */
//参数:net:BP网络
//     eta:学习速率
//     momentum:学习动量因子
//     e0:输出层误差
//     eh:隐藏层误差
void bpnn_train(BPNN *net, double eta, double momentum, double *eo, double *eh)
{
	int in, hid, out;
	double out_err, hid_err;

	in = net->input_n;
	hid = net->hidden_n;
	out = net->output_n;

	/*** 前向输入激活 ***/
	bpnn_layerforward(net->input_units, net->hidden_units, net->input_weights, in, hid);
	bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);

	/*** 计算隐藏层和输出层误差 ***/
	bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
	bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);
	*eo = out_err;
	*eh = hid_err;

	/*** 调整输入层和隐藏层权值 ***/
	bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_pre_weights, eta, momentum);
	bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_pre_weights, eta, momentum);
}

/* 保存BP网络 */
//参数：net：待保存的网络
//      filename:文件名
void bpnn_save(BPNN *net, char *filename)
{
	int n1, n2, n3, i, j, memont;
	double dvalue, **w;
	char *mem;
	FILE *fd;

	if ((fd = fopen(filename, "w")) == NULL)
	{
		printf("BPNN_SAVE: Cannnot create '%s'\n", filename);
		return;
	}

	n1 = net->input_n; n2 = net->hidden_n; n3 = net->output_n;
	printf("Saving %dx%dx%d network to '%s'\n", n1, n2, n3, filename);
	fflush(stdout);

	fwrite((char *) &n1, sizeof(int), 1, fd);
	fwrite((char *) &n2, sizeof(int), 1, fd);
	fwrite((char *) &n3, sizeof(int), 1, fd);

	memont = 0;
	w = net->input_weights;
	mem = (char *)malloc((unsigned)((n1 + 1)*(n2 + 1)*sizeof(double)));
	for (i = 0; i <= n1; i++)
	{
		for (j = 0; j <= n2; j++)
		{
			dvalue = w[i][j];
			fastcopy(&mem[memont], &dvalue, sizeof(double));
			memont += sizeof(double);
		}
	}

	fwrite(mem, (n1 + 1)*(n2 + 1)*sizeof(double), 1, fd);
	free(mem);

	memont = 0;
	w = net->hidden_weights;
	mem = (char *)malloc((unsigned)(n2 + 1)*(n3 + 1)*sizeof(double));
	for (i = 0; i <= n2; i++)
	{
		for (j = 0; j <= n3; j++)
		{
			dvalue = w[i][j];
			fastcopy(&mem[memont], &dvalue, sizeof(double));
			memont += sizeof(double);
		}
	}

	fwrite(mem, (n2 + 1)*(n3 + 1)*sizeof(double), 1, fd);
	free(mem);

	fclose(fd);
	return;
}

/* 从文件中读取BP网络 */
//参数：filename:输入的文件名
//返回：BP网络结构
BPNN *bpnn_read(char *filename)
{
	char *mem;
	BPNN *new1;
	int n1, n2, n3, i, j, memont;
	FILE *fd;

	if ((fd = fopen(filename, "r")) == NULL)
	{
		return (NULL);
	}

	printf("Reaidng '%s'\n", filename); fflush(stdout);

	fread((char *) &n1, sizeof(int), 1, fd);
	fread((char *) &n2, sizeof(int), 1, fd);
	fread((char *) &n3, sizeof(int), 1, fd);

	new1 = bpnn_internal_create(n1, n2, n3);

	printf("'%s' contains a %dx%dx%d network\n", filename, n1, n2, n3);
	printf("Reading input weights..."); fflush(stdout);

	memont = 0;
	mem = (char *)sizeof((unsigned)((n1 + 1)*(n2 + 1)*sizeof(double)));

	fread(mem, (n1 + 1)*(n2 + 1)*sizeof(double), 1, fd);
	for (i = 0; i <= n1; i++)
	{
		for (j = 0; j <= n2; j++)
		{
			fastcopy(&(new1->input_weights[i][j]), &mem[memont], sizeof(double));
			memont += sizeof(double);
		}
	}
	free(mem);

	printf("Done\nReading hidden weights..."); fflush(stdout);

	memont = 0;
	mem = (char *)malloc((unsigned)((n2 + 1)*(n3 + 1)*sizeof(double)));

	fread(mem, (n2 + 1)*(n3 + 1)*sizeof(double), 1, fd);
	for (i = 0; i <= n2; i++)
	{
		for (j = 0; j <= n3; j++)
		{
			fastcopy(&(new1->hidden_weights[i][j]), &mem[memont], sizeof(double));
			memont += sizeof(double);
		}
	}
	free(mem);
	fclose(fd);

	printf("Done\n"); fflush(stdout);

	bpnn_zero_weights(new1->input_pre_weights, n1, n2);
	bpnn_zero_weights(new1->hidden_pre_weights, n2, n3);

	return (new1);
}

