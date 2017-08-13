// 主程序

#include <iostream>
#include "bpnet.h"
using namespace std;

#define BPN_ITER 2000 // 迭代次数

void main()
{
	CBPNet bp;

	for (int i = 0; i < BPN_ITER; i++)
	{
		bp.Train(0, 0, 0);
		bp.Train(0, 1, 1);
		bp.Train(1, 0, 1);
		bp.Train(1, 0, 1);
		bp.Train(1, 1, 0);
	}
	cout << "0, 0 = " << bp.Run(0, 0) << endl;
	cout << "0, 1 = " << bp.Run(0, 1) << endl;
	cout << "1, 0 = " << bp.Run(1, 0) << endl;
	cout << "1, 1 = " << bp.Run(1, 1) << endl;

	cin.get();
}