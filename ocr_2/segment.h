
#ifndef _SEGMENT_H_
#define _SEGMENT_H_

#include<vector>
#include<opencv2\opencv.hpp>
using namespace std;
//ʵ�ַָ�ĺ���
/*
*���� filepath ���鵥·��
*��� vvM ��ά�ֿ�����
* ����ֵ 0����������
*/
int segment(const char* filepath, vector<vector<cv::Mat>>& vvM);
#endif