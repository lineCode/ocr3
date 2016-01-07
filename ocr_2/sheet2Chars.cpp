#include"sheet2Chars.h"
#include"predict.h"
using namespace cv;
//#define SAVEVMCHAR
/*
* private 过期
*
* 调用者：cutBody
* 作用：获得端点坐标
* author: zhangzhen
*/
int getPoint(vector < cv::Point >  &vpoint, cv::Point &p1, cv::Point &p2);


/*
* private
*
* 调用者：cutBody
* 作用：lsd检测直线
* author: zhangzhen
*/
void detectLines(const cv::Mat& image, vector<Vec4f>& lines_std);

/*
* private
*
* 调用者：cutBody
* 作用：统计主方向 ,返回值为主方向（单位：°）
* author: zhangzhen
*/
float countDirection(vector<Vec4f>& lines_std);


/*
*
* private
*
* 调用者：cutBody
* 作用：根据角度旋转图像
* author: zhangzhen
*/
void rotateImage(cv::Mat& image, float angle);

/*
*
* private
*
* 调用者：cutBody
* 作用：筛选直线 （第一步：删除不合格的直线，第二步，合并直线）
* author: zhangzhen
*/
//void filtLines(vector<Vec4f>& lines_std, vector<Vec4f>& lines);
void filtLines(vector<Vec4f>& lines_std, vector<Vec4f>& finallines, float disth = 30.0f, float combineth = 15.0f);
/*
*
* private
*
* 调用者：cutBody
* 作用：定位“中心”直线，并返回其标号
* author: zhangzhen
*/
int findCenterLine(vector<Vec4f>& lines);


/*
* private
* 基于最大相关度（Correlation）来计算阈值
* 调用者：analysisLayout
* 作用：使用Yen方法选择二值化阈值 ,返回值为 一个整数阈值
* author: Ding pengli
*/
int Yen(double* daHistogram, int NUM_GRAY = 256);

/*
* private
* 
* 调用者：segmentChars
* 作用：把图像块切成行块
* author: zhangzhen xuewenyuan
*/
int cut2Lines(const cv::Mat& vmroiitem, vector<cv::Mat>& vmlines);

/*
* private
*
* 调用者：segmentChars getROIItems
* 作用：把行块切成字符块
* author: zhangzhen xuewenyuan
*/
int cut2Chars(const cv::Mat& vmlines, vector<cv::Mat>& vmchars);


int pos = 0;//记录中心线坐标

int cutBody(const char * imagepath, cv::Mat &body)
{

	Mat image = imread(imagepath, IMREAD_GRAYSCALE);
	const int NORMALSIZE = 1000; //缩放后图像的尺寸
	float normalscale = 1.0f; //缩放因子
	Mat imagenormal;
	if (image.cols > NORMALSIZE)
	{
		resize(image, imagenormal, Size(NORMALSIZE, NORMALSIZE * image.rows / image.cols), 0.0, 0.0, INTER_AREA);
		normalscale = 1.0*image.cols / NORMALSIZE;
	}
	else
	{
		imagenormal = image.clone();
	}
	vector<Vec4f> lines_std;
	vector<Vec4f> lines;
	detectLines(imagenormal, lines_std);
	//主方向 angle
	float angle = countDirection(lines_std);
	//angle *= 2;
	//旋转图像
	rotateImage(image, angle);
	rotateImage(imagenormal, angle);
	lines_std.clear();
	//对矫正后的图像，提取直线
	detectLines(imagenormal, lines_std);
	filtLines(lines_std, lines);
	int ind = 0;//中心线标号
	ind=findCenterLine(lines);
	if (ind == -1)
	{
		return -1;
	}
	
	int rowrangestart = (int)(min(lines[ind - 1][1], lines[ind - 1][3])*normalscale);
	int rowrangeend = (int)(min(lines[ind + 1][1], lines[ind + 1][3])*normalscale);
	int colrangestart = (int)(min(lines[ind - 1][0] , lines[ind + 1][0]) * normalscale);
	int colrangeend = (int)(max(lines[ind - 1][2] , lines[ind + 1][2])  * normalscale);
	body = image.rowRange(rowrangestart, rowrangeend).colRange(colrangestart, colrangeend);


	int centerrow = (int)(min(lines[ind][1], lines[ind][3])*normalscale);
	pos = centerrow - rowrangestart;
	return 0;

	/*
	//zhangzhen begin
	Mat image = imread(imagepath, IMREAD_GRAYSCALE);
	if (!image.data)
	{
	cerr << "图像路径错误" << endl;
	return -1;
	}
	Mat imagebak = image.clone();
	Mat image2 = image.clone();
	Sobel(image, image, CV_8U, 1, 0, 3, 1, 0);
	Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(20, 2));
	morphologyEx(image, image, CV_MOP_DILATE, element);
	Sobel(image2, image2, CV_8U, 0, 1, 3, 1, 0);
	Mat m3 = image2*1.5 - image*0.5;
	threshold(m3, m3, 50, 255, CV_THRESH_BINARY);
	vector<vector<cv::Point>> contours;
	vector<vector<cv::Point>> linecontours;
	findContours(m3, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	vector<int> vi;
	for (vector<vector<cv::Point>>::iterator it = contours.begin(); it != contours.end();it++)
	{
	if (it->size() >= m3.cols * 3 / 4)
	{
	vi.push_back(it->at(0).y);
	linecontours.push_back(*it);
	}
	}
	int downlineindex = 0;//下直线索引
	int uplineindex = 0;//上直线索引
	if (vi.size() < 2)
	{
	cerr << "定位直线失败"<<endl;
	return -1;
	}
	if (vi.size() == 2)
	{
	downlineindex = 0;
	uplineindex = 1;
	}
	else
	{
	int mlen = 0;
	int mindex = 0;
	for (int i = 1; i < vi.size(); i++)
	{
	if (vi[i - 1] - vi[i]>mlen)
	{
	mlen = vi[i-1] - vi[i];
	mindex = i - 1;
	}
	}
	downlineindex = mindex;
	if (vi.size() -1 - mindex >= 2) //上半部分直线数大于2
	{
	uplineindex = mindex + 2;
	}
	else
	{
	uplineindex = mindex + 1;
	}
	}
	Point pointup1, pointup2;
	Point pointdown1, pointdown2;

	getPoint(linecontours[uplineindex], pointup1, pointup2);
	getPoint(linecontours[downlineindex], pointdown1, pointdown2);
	//未进行矫正， 可以通过pointup1, pointup2计算出斜率。
	body = imagebak.rowRange(min(pointup1.y, pointup2.y), max(pointdown1.y, pointdown2.y)).colRange(min(pointup1.x, pointdown1.x), max(pointup2.x, pointdown2.x));
	return 0;
	//zhangzhen end
	*/

	/*
	//wenyuan begin
	Mat input = imread(imagepath);
	Mat gray;

	if (input.channels() == 3)
	{
		cvtColor(input, gray, CV_BGR2GRAY);
	}
	else
	{
		gray = input;
	}


	// ！矫正
	Mat cvSobel;
	Sobel(gray, cvSobel, CV_8U, 0, 1, 3, 1, 0);
	vector<Vec4i> lines;
	HoughLinesP(cvSobel, lines, 1, CV_PI / 180, 100, 80, 1);
	Mat draw;
	input.copyTo(draw);
	Vec4i maxLong;
	double tempLong = 0;
	double angel = 0;
	for (int i = (int)lines.size() - 1; i >= 0; i--) {

		double temp = pow(lines[i][0] - lines[i][2], 2) + pow(lines[i][1] - lines[i][3], 2);
		double tmpAngel = atan((lines[i][1] - lines[i][3])*0.1 / ((lines[i][0] - lines[i][2])*0.1));
		if (temp > tempLong) {
			tempLong = temp;
			angel = tmpAngel;
			//maxLong = lines[i];

		}
		//画直线
		line(draw, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 2, 8, 0);
	}
	//imwrite("/Users/setsufumimoto/Desktop/test/cvlines.jpg", draw);
	//cout << "旋转角度：" << angel << " " << angel*180/CV_PI << endl;
	angel = angel < CV_PI / 2 ? angel : angel - CV_PI;
	if (angel != CV_PI / 2){
		double angelT = input.rows*tan(angel) / input.cols;
		angel = atan(angelT);
	}
	double angelD = angel * 180 / (float)CV_PI;
	Point2f center = Point2f(input.cols / 2, input.rows / 2);
	Mat rotateMat = getRotationMatrix2D(center, angelD, 1.0);
	Mat rotateImg;
	warpAffine(gray, rotateImg, rotateMat, input.size(), 1, 0, cvScalarAll(255));
	//imwrite("/Users/setsufumimoto/Desktop/test/rotateImg.jpg", rotateImg);

	// 矫正!

	//! [bin]
	// Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
	Mat bw;
	adaptiveThreshold(~rotateImg, bw, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);
	//imwrite("/Users/setsufumimoto/Desktop/test/bw.jpg", bw);

	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat out;
	dilate(bw, out, element);

	//! [bin]

	//! [init]
	// Create the images that will use to extract the horizontal and vertical lines
	Mat horizontal = out.clone();
	//! [init]

	//! [horiz]
	// Specify size on horizontal axis
	int horizontalsize = horizontal.cols / 20;

	// Create structure element for extracting horizontal lines through morphology operations
	Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontalsize, 3));

	// Apply morphology operations
	erode(horizontal, horizontal, horizontalStructure, Point(-1, -1));
	dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1));

	// Show extracted horizontal lines
	//imshow("horizontal", horizontal);
	//imwrite("/Users/setsufumimoto/Desktop/test/horizontal.JPG", horizontal);
	//! [horiz]
	//! 搜索直线
	vector<vector<cv::Point>> contours;
	findContours(horizontal, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	if (contours.size() < 3) {
		cout << "Failure! Reaon : Can't detect the table.(table lines error)" << endl;
		return -1;
	}

	vector<pair<Point, Point>> Up;//保存表头的两条直线
	vector<pair<Point, Point>> Down;//保存表尾的一条直线
	for (int i = 0; i < contours.size(); i++)  {
		if (contours.size() < 3) {
			break;
		}
		int Lx = input.cols, Rx = 0;
		pair<Point, Point> LR;
		for (int j = 0; j< contours[i].size(); j++) {
			if (contours[i][j].x < Lx) {
				Lx = contours[i][j].x;
				LR.first = contours[i][j];
			}
			if (contours[i][j].x > Rx) {
				Rx = contours[i][j].x;
				LR.second = contours[i][j];
			}
		}
		if (LR.first.y > input.rows / 2 && LR.second.x - LR.first.x > input.cols / 2) {
			if (Down.empty()) Down.push_back(LR);
			else if (LR.first.y < Down[0].first.y) {
				Down.pop_back();
				Down.push_back(LR);
			}
		}
		else if (LR.first.y < input.rows / 2 && LR.second.x - LR.first.x>input.cols / 2){
			if (Up.empty()) Up.push_back(LR);
			else if (Up.size() == 1) {
				if (LR.first.y > Up[0].first.y)
					Up.insert(Up.begin(), LR);
				else Up.push_back(LR);
			}
			else if (Up.size() == 2) {

				if (LR.first.y > Up[0].first.y) {
					Up.insert(Up.begin(), LR);
					Up.pop_back();
				}
				else if (LR.first.y > Up[1].first.y) {
					Up.insert(Up.begin() + 1, LR);
					Up.pop_back();
				}
			}
		}
	}

	if (Up.size() < 2 || Down.empty()) {
		cerr << "Failure! Reaon : Can't detect the table.(table lines error)" << endl;
		return -1;
	}
	Mat draw2;
	rotateImg.copyTo(draw2);
	line(draw2, Down[0].first, Down[0].second, Scalar(0, 255, 255), 2, 8, 0);
	line(draw2, Up[0].first, Up[0].second, Scalar(0, 255, 255), 2, 8, 0);
	line(draw2, Up[1].first, Up[1].second, Scalar(0, 255, 255), 2, 8, 0);
	//imwrite("/Users/setsufumimoto/Desktop/test/drawLines.JPG", draw2);
	//imshow("drawLines", draw2);

	//! 搜索直线
	//! 仿射变换
	vector<Point2f> srcPoint(4);
	vector<Point2f> dstPoint(4);

	srcPoint[0] = Point2f(Up[1].first.x, Up[1].first.y);
	srcPoint[1] = Point2f(Up[1].second.x, Up[1].second.y);
	srcPoint[2] = Point2f(Down[0].first.x, Down[0].first.y);
	srcPoint[3] = Point2f(Down[0].second.x, Down[0].second.y);

	int h = ((Down[0].first.y - Up[1].first.y) + (Down[0].second.y - Up[1].second.y)) / 2;
	int w = ((Down[0].second.x - Down[0].first.x) + (Up[1].second.x - Up[1].first.x)) / 2;

	body = Mat::zeros(h, w, input.type());

	dstPoint[0] = Point2f(0, 0);
	dstPoint[1] = Point2f(body.cols - 1, 0);
	dstPoint[2] = Point2f(0, body.rows - 1);
	dstPoint[3] = Point2f(body.cols - 1, body.rows - 1);
	Mat transmtx = getPerspectiveTransform(srcPoint, dstPoint);

	warpPerspective(rotateImg, body, transmtx, body.size());
	//imwrite("/Users/setsufumimoto/Desktop/test/dst.JPG", dst);
	//imshow("dst", dst);
	//! 仿射变换

	return 0;

	//wenyuan end
	*/
}




int analysisLayout(const cv::Mat& body, vector<cv::Mat>& vmitems)
{
	///*
	//zhangzhen begin
	const int MaxCols = 750; // 缩放尺寸
	const float SIGMA_DETA_FACTOR = 0.35; //特征值 均值与方差的调节因子
	const int MEDIANBLUR_SIZE = 9; // 中值滤波器模板尺寸
	Mat body2;
	float body2scale = 1.0f;
	if (body.cols > MaxCols)
	{
		resize(body, body2, Size(MaxCols, MaxCols * body.rows / body.cols), 0.0, 0.0, INTER_AREA);
		body2scale = (float)(1.0*body.cols / MaxCols);
	}
	else
	{
		body2 = body.clone();
	}

	//adaptiveThreshold(body2, body2, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 7, -2);

	//统计直方图，并归一化
	double hist[256] = { 0 };
	for (int j = 0; j < body2.rows; j++)
	{
		uchar* data = body2.ptr<uchar>(j);
		for (int i = 0; i <body2.cols; i++)
		{
			hist[data[i]] += 1;
		}
	}
	for (int i = 0; i < 256; i++)
	{
		hist[i] /= (body2.rows*body2.cols);
	}
	//使用Yen选择阈值
	threshold(body2, body2, Yen(hist,256), 255, THRESH_BINARY);
	Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(8, 2));//Size(8,2) 经验值
	morphologyEx(body2, body2, CV_MOP_ERODE, element);
	int cols = body2.cols;
	int partrows = body2.rows / 2;
	vector<float> feature(cols);
	for (int ic = 0; ic < cols; ic++)
	{
		cv::Mat tempdst = body2.col(ic).rowRange(15, partrows);
		cv::Scalar meanv, stdDev;
		cv::meanStdDev(tempdst, meanv, stdDev);
		feature[ic] = (float)(min(10.0,meanv[0] / (stdDev[0] + 1)));
	}
	Mat featureMat(feature);
	cv::Scalar meanv, stdDev;
	cv::meanStdDev(featureMat, meanv, stdDev);
	float thd = (float)(meanv[0] + SIGMA_DETA_FACTOR*(stdDev[0]));
	featureMat = featureMat > thd;
	cv::medianBlur(featureMat, featureMat, MEDIANBLUR_SIZE);


	//分割项目
	int ibegin = 0, jbegin = 0;
	int iend = 0, jend = 0;
	int ic = 0;
	//while (ic < cols && featureMat.ptr<uchar>(ic)[0] == 0)
	//{
	//	ic++;
	//}
	//ibegin = ic;
	//while (ic < cols &&featureMat.ptr<uchar>(ic)[0] == 255)
	//{
	//	ic++;
	//}
	//jbegin = ic - 1;
	while (ic < cols)
	{

		while (ic < cols &&featureMat.ptr<uchar>(ic)[0] == 0)
		{
			ic++;
		}
		iend = ic;
		while (ic < cols &&featureMat.ptr<uchar>(ic)[0] == 255)
		{
			ic++;
		}
		jend = ic - 1;
		if (jend < iend) // 以黑色区域结束 
		{
			break;
		}
		Mat item = body.colRange((int)(body2scale*(ibegin + jbegin) / 2), (int)(body2scale*(iend + jend) / 2));
		ibegin = iend;
		jbegin = jend;
		if (item.cols>0)
			vmitems.push_back(item);
	}

	//zhangzhen end
	//*/

	///*
	//wenyuan begin
	//wenyuan end
	//*/
	return 0;
}

bool isROIItem(std::vector<int>& viLabel)
{
	char num2str[64] = { 0 };
	for (int i = 0; i < viLabel.size(); i++)
	{
		sprintf(num2str, "%s%d", num2str, viLabel[i]);
	}
	string str(num2str);
	if (str.find("56") != string::npos || str.find("78") != string::npos || str.find("1112") != string::npos)
		return true;
	return false;
}
int getROIItems(const vector<cv::Mat>& vmitems, vector<cv::Mat>& vmroiitems)
{
	//调用
	//int predict(const std::vector<cv::Mat>& vM, std::vector<int>& vvi)
	//根据结果判断哪个项目块为感兴趣的。


	for (int i = 0; i < vmitems.size(); i++)
	{
		//vector<cv::Mat> vmlines;

		cv::Mat header = vmitems[i].rowRange(0, pos+2); //pos 全局变量，在cutbody时记录值

		//精确定位表头行,原理等同于切割行，但是具体参数一样
		//Sobel(header, header, CV_8U, 1, 0, 3, 1, 0);

		const float SIGMA_DETA_FACTOR = 0.3;
		int rows = header.rows;
		int cols = header.cols;
		vector<float> feature(rows);
		for (int ir = 0; ir < rows; ir++)
		{
			cv::Mat tempdst = header.row(ir);
			cv::Scalar meanv, stdDev;
			cv::meanStdDev(tempdst, meanv, stdDev);
			feature[ir] = (float)(min(18.0, meanv[0] / (stdDev[0] + 1)));
		}
		Mat featureMat(feature);
		cv::Scalar meanv, stdDev;
		cv::meanStdDev(featureMat, meanv, stdDev);
		float thd = (float)(meanv[0] + SIGMA_DETA_FACTOR*(stdDev[0]));
		featureMat = featureMat > thd;

		int up = 0, down = 0;
		int half = rows / 2;
		while (half >= 0 && featureMat.ptr<uchar>(half)[0] == 0)
		{
			half--;
		}
		up = half;// max(0, half - 1);
		half = rows / 2;
		while (half < rows && featureMat.ptr<uchar>(half)[0] == 0)
		{
			half++;
		}
		down = min(rows, half + 1);	
		if ((down - up + 3) < rows / 2 || (down - up + 1)>rows) //没有文字
			continue;
		std::vector<cv::Mat> vmhead;
		std::vector<cv::Mat> vMChar;
		std::vector<int> viLabel;
		cut2Chars(header.rowRange(up,down), vMChar);
		predict(vMChar, viLabel);

#ifdef SAVEVMCHAR
		for (int i=0;i<vMChar.size();i++)
		{
			char impath[64] = { 0 };
			sprintf(impath, "../Output/%d.png", i + 100);
			imwrite(impath, vMChar[i]);
		}
		
#endif
		
		if (isROIItem(viLabel))
			vmroiitems.push_back(vmitems[i].rowRange(pos + 2, vmitems[i].rows));
	}


	//检验item svm
	//std::vector<cv::Mat> vM;
	//std::vector<int> vi;
	//cv::Mat m1 = imread("..\\Input\\1ce.png");
	//cv::Mat m2 = imread("..\\Input\\2mu.png");
	//cv::Mat m3 = imread("..\\Input\\3dai.png");
	//cv::Mat m4 = imread("..\\Input\\4hao.png");
	//cv::Mat m5 = imread("..\\Input\\5xiang.png");
	//cv::Mat m6 = imread("..\\Input\\6mu.png");
	//vM.push_back(m1);
	//vM.push_back(m2);
	//vM.push_back(m3);
	//vM.push_back(m4);
	//vM.push_back(m5);
	//vM.push_back(m6);
	//predict(vM, vi);

	return 0;
}

//过期
int cutLines(const vector<cv::Mat>& vmroiitems, vector<cv::Mat>& vmlines)
{
	return 0;
}




/*//暂定接口，这步比较复杂
* 输入： vmlines 行块。
* 输出：vvmchars 字符图像块（二维数组）
* 放回值： 0 代表正常
* vvmchars格式要求：	项目1字1字2字3 ..
*						结果1字1字2....
*						项目2字1字2字3 ..
*						结果2字1字2....
*						....
*
*/
int segmentChars(const vector<cv::Mat>& vmroiitems, vector<vector<cv::Mat>> &vvmchars)
{
	//调用
	//int predict(const cv::Mat& charMat);
	//识别单个字。
	if (vmroiitems.size() % 2 == 1)
	{
		return -1;
	}
	int flag = -1;
	for (size_t i = 1; i < vmroiitems.size(); i+=2)
	{
		vector<cv::Mat> vmlines;//测试项目
		vector<cv::Mat> vmlines2;//测试结果
		cut2Lines(vmroiitems[i-1], vmlines);
		cut2Lines(vmroiitems[i], vmlines2);
		if (vmlines.size() != vmlines2.size())//测试项目和结果行数不一致
		{
			return flag;
		}
		flag = 0;
		for (size_t iline = 0; iline < vmlines.size(); iline++)
		{
			vector<cv::Mat> vmtestchars;//测试项目行（S）
			vector<cv::Mat> vmresultchars;//测试结果行(S)
			cut2Chars(vmlines[iline], vmtestchars);
			cut2Chars(vmlines2[iline], vmresultchars);
			vvmchars.push_back(vmtestchars);
			vvmchars.push_back(vmresultchars);
		}
			

	}
	return 0;
}


int getPoint(vector < cv::Point >  &vpoint, cv::Point &p1, cv::Point &p2)
{
	if (vpoint.size() <= 1)
		return -1;
	p1 = vpoint[0];
	p2 = vpoint[0];
	for (int i = 0; i < vpoint.size(); i++)
	{
		if (p1.x > vpoint[i].x)
		{
			p1 = vpoint[i];
		}
		if (p2.x < vpoint[i].x)
		{
			p2 = vpoint[i];
		}

	}
	return 0;
}

void detectLines(const cv::Mat& image, vector<Vec4f>& lines_std)
{
	/*
	_refine 精度
	_scale 缩放尺度
	_sigma_scal 高斯核 sigma
	_quant 梯度
	_ang_th 角度
	_log_eps
	_density_th 密度
	_n_bins 箱数
	*/
	Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_STD, 1, 0.6, 1.5, 10, 0.0, 0.7);
	ls->detect(image, lines_std);
}

float countDirection(vector<Vec4f>& lines_std)
{
	const float DISTH = 50.f;
	const float DIFFANGLE = 0.2;//主方向阈值
	float kangle = 0;
	int kcount = 0;
	// 只统计在body前2/3区域的直线，这样是否合理？对目前的图片有作用。
	for (size_t i = lines_std.size()*2/3; i > 0; i--)
	{
		float f1x = lines_std[i][0];
		float f1y = lines_std[i][1];
		float f2x = lines_std[i][2];
		float f2y = lines_std[i][3];
		if (pow(pow(f2x - f1x, 2) + pow(f2y - f1y, 2), 0.5) > DISTH)
		{
			float tmpk = (f2y - f1y) / (f2x - f1x);
			if (abs(kangle - tmpk) < DIFFANGLE)
			{
				kangle = (kangle*kcount + tmpk) / (kcount + 1);
				kcount += 1;
			}
			else
			{
				kcount--;
			}
			if (kcount <= 0)
			{
				kangle = tmpk;
				kcount = 1;
			}
		}


	}
	return atanf(kangle) * 180 / CV_PI;
}

void rotateImage(cv::Mat& image, float angle)
{
	Point2f center = Point2f(image.cols / 2, image.rows / 2);
	Mat rotateMat = getRotationMatrix2D(center,angle, 1.0);
	warpAffine(image, image, rotateMat, image.size(), 1, 0, cvScalarAll(255));
}

void filtLines(vector<Vec4f>& lines_std, vector<Vec4f>& finallines, float disth, float combineth)
{
	vector<Vec4f> lines_long;
	const float DIS = disth;
	const float ANG = tanf(2.0f*CV_PI / 180);
	for (size_t i = 0; i < lines_std.size(); i++)
	{
		float f1x = lines_std[i][0];
		float f1y = lines_std[i][1];
		float f2x = lines_std[i][2];
		float f2y = lines_std[i][3];
		Vec2f tv;
		tv[0] = (f2y - f1y) / (f2x - f1x);
		tv[1] = tv[0] * (-f1x) + f1y;
		//k_b.push_back(tv);
		float distance = pow(pow(f2x - f1x, 2) + pow(f2y - f1y, 2), 0.5);
		float angle = (f2y - f1y) / (f2x - f1x);
		if (distance > DIS && abs(angle) < ANG)
		{
			lines_long.push_back(lines_std[i]);
			//cout << distance << endl;
		}

	}
	const float COMBINETH = combineth;
	vector<bool>flagvisit(lines_long.size(), false);

	for (int i = 0; i < lines_long.size();)
	{
		if (lines_long[i][0]>lines_long[i][2])
		{
			Vec4f tmpswqp = lines_long[i];
			lines_long[i][0] = tmpswqp[2];
			lines_long[i][1] = tmpswqp[3];
			lines_long[i][2] = tmpswqp[0];
			lines_long[i][3] = tmpswqp[1];

		}
		Vec4f tempv = lines_long[i];
		Vec2f k_b;
		k_b[0] = (tempv[3] - tempv[1]) / (tempv[2] - tempv[0]);//k
		k_b[1] = -k_b[0] * tempv[0] + tempv[1];//b
		int j = i + 1;
		for (; j < lines_long.size(); j++)
		{
			float dis = abs(k_b[0] * lines_long[j][0] + k_b[1] - lines_long[j][1]) / pow(1 + pow(k_b[0], 2), 0.5);
			if (dis > COMBINETH)
			{
				break;
			}
			if (lines_long[j][0]>lines_long[j][2])
			{
				Vec4f tmpswqp = lines_long[j];
				lines_long[j][0] = tmpswqp[2];
				lines_long[j][1] = tmpswqp[3];
				lines_long[j][2] = tmpswqp[0];
				lines_long[j][3] = tmpswqp[1];

			}
			tempv[1] = tempv[0]<lines_long[j][0] ? tempv[1] : lines_long[j][1];//谁在左边取谁的x坐标
			tempv[3] = tempv[2]>lines_long[j][2] ? tempv[3] : lines_long[j][3];//谁在右边取谁的y坐标
			tempv[0] = min(tempv[0], lines_long[j][0]);
			tempv[2] = max(tempv[2], lines_long[j][2]);

		}

		finallines.push_back(tempv);
		i = j;
	}
}

int findCenterLine(vector<Vec4f>& lines)
{
	float maxwidth = 0.0f;
	int ind = 0;
	for (size_t i = 1; i < lines.size(); i++)
	{
		float prey = (lines[i - 1][1] + lines[i - 1][3]) / 2;
		float cury = (lines[i][1] + lines[i][3]) / 2;
		if (cury - prey>maxwidth)
		{
			maxwidth = cury - prey;
			ind = i - 1;
		}
	}

	if (ind<1 || ind + 1>lines.size())
	{
		cerr << "没有中心线" << endl;
		return -1;
	}
	return ind;
}

int Yen(double *daHistogram,int NUM_GRAY)
{
	int MAX_GRAY = NUM_GRAY - 1;
	int iThreshold = 0, i;
	double *P1;
	double *P1_sq;
	double *P2_sq;

	//计算累计归一化直方图
	P1 = new  double[NUM_GRAY];
	P1[0] = daHistogram[0];
	for (i = 1; i < NUM_GRAY; i++)
	{
		P1[i] = P1[i - 1] + daHistogram[i];
	}

	P1_sq = new double[NUM_GRAY];
	P1_sq[0] = daHistogram[0] * daHistogram[0];
	for (i = 1; i < NUM_GRAY; i++)
	{
		P1_sq[i] = P1_sq[i - 1] + daHistogram[i] * daHistogram[i];
	}

	P2_sq = new double[NUM_GRAY];
	P2_sq[MAX_GRAY] = daHistogram[MAX_GRAY] * daHistogram[MAX_GRAY];
	for (i = MAX_GRAY - 1; i >= 0; i--)
	{
		P2_sq[i] = P2_sq[i + 1] + daHistogram[i] * daHistogram[i];
	}

	// 定位直方图的峰顶位置
	int iIndEnd = 0;
	for (i = 0; i<NUM_GRAY; i++)
	{
		if (daHistogram[i] > daHistogram[iIndEnd])
			iIndEnd = i;
	}
	if (iIndEnd < 10)
		iIndEnd = NUM_GRAY - 1;

	// 搜索最优阈值，阈值范围不超过 iIndEnd，使得阈值保持在较低值的位置
	double dCrit = 0.0;
	double dMaxCrit = DBL_MIN;
	for (i = 0; i < iIndEnd - 1; i++)
	{
#define SAFE_LOG( x ) ( ( ( x ) > 0.0 ) ? log ( ( x ) ) : ( 0.0 ) )
		dCrit = -1.0 * SAFE_LOG(P1_sq[i] * P2_sq[i + 1]) +
			2 * SAFE_LOG(P1[i] * (1.0 - P1[i]));
		if (dCrit > dMaxCrit)
		{
			dMaxCrit = dCrit;
			iThreshold = i;
		}
	}

	delete[] P1;
	delete[] P1_sq;
	delete[] P2_sq;

	return iThreshold;
}


int cut2Lines(const cv::Mat& vmroiitem, vector<cv::Mat>& vmlines)
{
	const float SIGMA_DETA_FACTOR = 0.0;
	const int MEDIANBLUR_SIZE = 5;
	const float FEATHRETH = 30.0;
	int rows = vmroiitem.rows;
	int cols = vmroiitem.cols;
	vector<float> feature(rows);
	vector<float> tfeature;
	//float maxfeature = 0.0f;
	for (int ir = 0; ir < rows; ir++)
	{
		cv::Mat tempdst = vmroiitem.row(ir);
		cv::Scalar meanv, stdDev;
		cv::meanStdDev(tempdst, meanv, stdDev);
		feature[ir] = (float)(min(25.0, meanv[0] / (stdDev[0] + 1)));
		//maxfeature = max(maxfeature, feature[ir]);
		//float f = meanv[0] / (stdDev[0] + 1);
		//if (f<FEATHRETH)
		//{
		//	feature[ir] = f;
		//	tfeature.push_back(f);
		//}
		//else
		//{
		//	feature[ir] = FEATHRETH;
		//}
	}
	//for (int i = 0; i < feature.size(); i++)
	//{
	//	if (maxfeature - feature[i] >= 5.0f)
	//	{
	//		tfeature.push_back(feature[i]);
	//	}
	//}
	Mat featureMat(feature);
	//Mat tfeatureMat(tfeature);
	cv::Scalar meanv, stdDev;
	cv::meanStdDev(featureMat, meanv, stdDev);
	float thd = (float)(meanv[0] + SIGMA_DETA_FACTOR*(stdDev[0]));
	featureMat = featureMat > thd;
	//cv::medianBlur(featureMat, featureMat, MEDIANBLUR_SIZE);
	//统计平均字宽
	int maxcontinueblacklen = 0;
	int tmpbacklen = 0;
	//featureMat.ptr<uchar>(ic)[0]
	int fMros = featureMat.rows;
	int fMcols = featureMat.cols;
	for (int ir = 0; ir < fMros; ir++)
	{
		if (featureMat.ptr<uchar>(ir)[0] == 0)
		{
			tmpbacklen++;
		}
		else
		{
			tmpbacklen = 0;
		}
		maxcontinueblacklen = max(maxcontinueblacklen, tmpbacklen);
	}
	//int maxcontinuewhitelen = 0;
	int tmpwhitelen = 0;
	for (int ir = 0; ir < fMros; )
	{
		if (featureMat.ptr<uchar>(ir)[0] == 0)
		{
			int tir = min(ir + maxcontinueblacklen, fMros - 1);
			if (tir != fMros - 1&&featureMat.ptr<uchar>(tir)[0] == 0)//跳过“短”的黑色区域
			{
				while (ir<fMros&&featureMat.ptr<uchar>(ir)[0] == 0)
				{
					ir++;
				}
			}
			else//确定黑色区域边缘
			{
				while (tir<fMros&&featureMat.ptr<uchar>(tir)[0] == 255)
				{
					tir--;
				}
				vmlines.push_back(vmroiitem.rowRange(max(0,ir - 1), min(tir + 1,fMros)));
				ir = tir + 1;
			}
			tmpwhitelen = 0;
		}
		else
		{
			ir++;
			tmpwhitelen++;
		}
		//连续“白”区域
		if (tmpwhitelen > (int)(maxcontinueblacklen*1.5))
		{
			break;
		}
	}

	return 0;
}

int cut2Chars(const cv::Mat& mline, vector<cv::Mat>& vmchars)
{
	const float SIGMA_DETA_FACTOR = 0.75;
	const int MEDIANBLUR_SIZE = 1;
	int rows = mline.rows;
	int cols = mline.cols;
	vector<float> feature(cols);
	for (int ic = 0; ic < cols; ic++)
	{
		cv::Mat tempdst = mline.col(ic);
		cv::Scalar meanv, stdDev;
		cv::meanStdDev(tempdst, meanv, stdDev);
		feature[ic] = (float)(min(25.0, meanv[0] / (stdDev[0] + 1)));
	}
	vector<float> tfeature;
	for (int i = 0; i<feature.size(); i++)
	{
		if (feature[i] < 24.5)
			tfeature.push_back(feature[i]);
	}
	Mat tfeatureMat(tfeature);
	Mat featureMat(feature);
	cv::Scalar meanv, stdDev;
	cv::meanStdDev(tfeatureMat, meanv, stdDev);
	float thd = (float)(meanv[0] + SIGMA_DETA_FACTOR*(stdDev[0]));
	featureMat = featureMat > thd;
	cv::medianBlur(featureMat, featureMat, MEDIANBLUR_SIZE);
	int begin = 0, end = 0;
	for (int i = 0; i < feature.size();)
	{
		if (featureMat.ptr<uchar>(i)[0] == 0)
		{
			begin = i;
			int j = i + 1;
			while (j < feature.size() && featureMat.ptr<uchar>(j)[0] == 0)
			{
				j++;
			}
			end = j;
			
			if (end - begin>(int)(mline.rows / 4.0))//字符宽度不应太小
			{
				vmchars.push_back(mline.colRange(begin, end));
			}
			i = j + 1;

			
		}
		else
		{
			i++;
		}
	}
	return 0;
}