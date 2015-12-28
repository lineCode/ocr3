
int cutBody(const char * imagepath, cv::Mat &body)
{
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
	HoughLinesP(cvSobel, lines, 1, CV_PI/180, 100, 80, 1);
	Mat draw;
	input.copyTo(draw);
	Vec4i maxLong;
	double tempLong = 0;
	double angel = 0;
	for(int i = (int)lines.size()-1; i >= 0; i --) {
		
		double temp = pow(lines[i][0]-lines[i][2], 2)+pow(lines[i][1]-lines[i][3], 2);
		double tmpAngel = atan((lines[i][1]-lines[i][3])*0.1/((lines[i][0]-lines[i][2])*0.1));
		if (temp > tempLong ) {
			tempLong = temp;
			angel = tmpAngel;
			//maxLong = lines[i];
			
		}
		//画直线
		line(draw, Point(lines[i][0],lines[i][1]), Point(lines[i][2],lines[i][3]), Scalar(0,0,255),2,8,0);
	}
	//imwrite("/Users/setsufumimoto/Desktop/test/cvlines.jpg", draw);
	//cout << "旋转角度：" << angel << " " << angel*180/CV_PI << endl;
	angel = angel<CV_PI/2 ? angel : angel-CV_PI;
	if(angel != CV_PI/2){
		double angelT = input.rows*tan(angel)/input.cols;
		angel = atan(angelT);
	}
	double angelD = angel*180/(float)CV_PI;
	Point2f center = Point2f(input.cols / 2, input.rows / 2);
	Mat rotateMat = getRotationMatrix2D(center, angelD, 1.0);
	Mat rotateImg;
	warpAffine(gray, rotateImg, rotateMat, input.size(), 1, 0,cvScalarAll(255));
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
	Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontalsize,3));
	
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
	for (int i = 0; i< contours.size(); i++)  {
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
		if (LR.first.y > input.rows/2 && LR.second.x-LR.first.x>input.cols/2) {
			if (Down.empty()) Down.push_back(LR);
			else if (LR.first.y < Down[0].first.y) {
				Down.pop_back();
				Down.push_back(LR);
			}
		}
		else if (LR.first.y < input.rows/2 && LR.second.x-LR.first.x>input.cols/2){
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
					Up.insert(Up.begin()+1, LR);
					Up.pop_back();
				}
			}
		}
	}
	
	if (Up.size() < 2 || Down.empty() ) {
		cerr << "Failure! Reaon : Can't detect the table.(table lines error)" << endl;
		return -1;
	}
	Mat draw2;
	rotateImg.copyTo(draw2);
	line(draw2,Down[0].first , Down[0].second, Scalar(0,255,255),2,8,0);
	line(draw2,Up[0].first , Up[0].second, Scalar(0,255,255),2,8,0);
	line(draw2,Up[1].first , Up[1].second, Scalar(0,255,255),2,8,0);
	//imwrite("/Users/setsufumimoto/Desktop/test/drawLines.JPG", draw2);
	//imshow("drawLines", draw2);
	
	//! 搜索直线
	//! 仿射变换
	vector<Point2f> srcPoint(4);
	vector<Point2f> dstPoint(4);
	
	srcPoint[0] = Point2f(Up[1].first.x,Up[1].first.y);
	srcPoint[1] = Point2f(Up[1].second.x,Up[1].second.y);
	srcPoint[2] = Point2f(Down[0].first.x,Down[0].first.y);
	srcPoint[3] = Point2f(Down[0].second.x,Down[0].second.y);
	
	int h = ((Down[0].first.y-Up[1].first.y)+(Down[0].second.y-Up[1].second.y))/2;
	int w = ((Down[0].second.x-Down[0].first.x)+(Up[1].second.x-Up[1].first.x))/2;
	
	body = Mat::zeros(h, w, input.type());
	
	dstPoint[0] = Point2f(0,0);
	dstPoint[1] = Point2f(body.cols-1,0);
	dstPoint[2] = Point2f(0,body.rows-1);
	dstPoint[3] = Point2f(body.cols-1,body.rows-1);
	Mat transmtx = getPerspectiveTransform(srcPoint, dstPoint);
	
	warpPerspective(rotateImg, body, transmtx, body.size());
	//imwrite("/Users/setsufumimoto/Desktop/test/dst.JPG", dst);
	//imshow("dst", dst);
	//! 仿射变换

	return 0;
}