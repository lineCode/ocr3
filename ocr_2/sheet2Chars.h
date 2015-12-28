#include<vector>
#include<opencv2\opencv.hpp>

using namespace std;


/*
* �и�鵥���岿�֣�������
* ���룺imagepath ͼ��·��
* �����body ���鵥���岿��
* ����ֵ��0 ��������
*/
int cutBody(const char * imagepath, cv::Mat &body);

/*
* �������
* ���룺body ���鵥���岿��
* �����vmitems ��Ŀ
* ����ֵ��0 ��������
*/
int analysisLayout(const cv::Mat& body, vector<cv::Mat>& vmitems);

/*
* ��ȡ����Ȥ����Ŀ������������Ŀ���С����������
* ���룺 vmitems ������Ŀ
* ����� vmroiitems ����Ȥ��Ŀ
* ����ֵ��0 ��������
*/
int getROIItems(const vector<cv::Mat>& vmitems, vector<cv::Mat>& vmroiitems);

/* ����
* ����Ŀͼ���ָ���п�
* ���룺 vmroiitems ��Ŀͼ���
* ����� vmlines �п顣
* ����ֵ��0 ��������
* vmlines�п��ʽҪ����Ŀ1�п� ���1�п� ��Ŀ2�п� ���2�п�....
*/
int cutLines(const vector<cv::Mat>& vmroiitems, vector<cv::Mat>& vmlines);


/*//�ݶ��ӿڣ��ⲽ�Ƚϸ���
* ���룺 vmlines �п顣
* �����vvmchars �ַ�ͼ��飨��ά���飩
* ����ֵ�� 0 ��������
* vvmchars��ʽҪ��	��Ŀ1��1��2��3 ..
*						���1��1��2....
*						��Ŀ2��1��2��3 ..
*						���2��1��2....
*						....
*
*/
int segmentChars(const vector<cv::Mat>& vmROIitems, vector<vector<cv::Mat>> &vvmchars);
