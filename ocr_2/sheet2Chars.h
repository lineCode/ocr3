#include<vector>
#include<opencv2\opencv.hpp>

using namespace std;


/*
* 切割化验单主体部分，并矫正
* 输入：imagepath 图像路径
* 输出：body 化验单主体部分
* 返回值：0 代表正常
*/
int cutBody(const char * imagepath, cv::Mat &body);

/*
* 版面分析
* 输入：body 化验单主体部分
* 输出：vmitems 项目
* 返回值：0 代表正常
*/
int analysisLayout(const cv::Mat& body, vector<cv::Mat>& vmitems);

/*
* 获取感兴趣的项目，即“测试项目”列、“结果”列
* 输入： vmitems 所有项目
* 输出： vmroiitems 感兴趣项目
* 返回值：0 代表正常
*/
int getROIItems(const vector<cv::Mat>& vmitems, vector<cv::Mat>& vmroiitems);

/* 过期
* 把项目图像块分割成行块
* 输入： vmroiitems 项目图像块
* 输出： vmlines 行块。
* 返回值：0 代表正常
* vmlines行块格式要求：项目1行块 结果1行块 项目2行块 结果2行块....
*/
int cutLines(const vector<cv::Mat>& vmroiitems, vector<cv::Mat>& vmlines);


/*//暂定接口，这步比较复杂
* 输入： vmlines 行块。
* 输出：vvmchars 字符图像块（二维数组）
* 返回值： 0 代表正常
* vvmchars格式要求：	项目1字1字2字3 ..
*						结果1字1字2....
*						项目2字1字2字3 ..
*						结果2字1字2....
*						....
*
*/
int segmentChars(const vector<cv::Mat>& vmROIitems, vector<vector<cv::Mat>> &vvmchars);
