#include <iostream>
#include <iomanip>
#include <fstream>
#include <ctime>
#include <cmath>

#include <unistd.h>
#include <cstdio>
#include <cstdlib>

#include <nicp/imageutils.h>
#include <nicp/pinholepointprojector.h>
#include <nicp/depthimageconverterintegralimage.h>
#include <nicp/statscalculatorintegralimage.h>
#include <nicp/alignerprojective.h>
#include <nicp/merger.h>
#include <nicp/cloud.h>
#include <nicp/pointprojector.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "epnp.h"
#include "RobustMatcher.h"

#define PI 3.1415926

using namespace std;
using namespace cv;

nicp::Cloud currentCloud;
nicp::Cloud previous_cloud;
nicp::Cloud nicp_complete_cloud;
nicp::Merger merger;
nicp::PinholePointProjector pointProjector;
nicp::StatsCalculatorIntegralImage statsCalculator;
nicp::PointInformationMatrixCalculator pointInformationMatrixCalculator;
nicp::NormalInformationMatrixCalculator normalInformationMatrixCalculator;
nicp::DepthImageConverterIntegralImage converter;
nicp::Linearizer linearizer;
nicp::AlignerProjective aligner;
nicp::CorrespondenceFinderProjective correspondenceFinder;

Mat previous_image;
Mat current_image;

Eigen::Matrix3f cameraMatrix;
Eigen::Matrix4f cow_transformation;

const float point_projector_min_distance = 0.1f;
const float point_projector_max_distance = 4.0f;

const float MERGE_MAX_DISTANCE = 0.2;

const int input_image_height = 320;
const int input_image_width = 480;

const int MAX_FEATURES = 800;
const int minHessian = 100;
const float GOOD_MATCH_PERCENT = 0.2;

const int RANSAC_MAX_ITERATIONS = 1000;
const float RANSAC_MIN_INLIERS = 0.7;
const double RANSAC_INLIER_DISTANCE = 0.22;

const int ICP_MAX_ITERATIONS = 7;
const float ICP_MAX_CORRESPONDECE_DISTANCE = 50.0f;

void initialize_matrixes(){
    cameraMatrix << 300.0f,   0.0f, 240.0f,
        0.0f, 300.0f, 160.0f,
        0.0f,   0.0f,   1.0f;

    Eigen::Matrix4f A,B,C;
    C << 1.0f, 0.0f, 0.0f, 0.0f,
         0.0f, 1.0f, 0.0f, 0.0f,
         0.0f, 0.0f, 1.0f,-2.0f,
         0.0f, 0.0f, 0.0f, 1.0f;

    B << cos(-20*PI/180),  0.0f, sin(-20*PI/180), 0.0f,
         0.0f,         1.0f, 0.0f,        0.0f,
         -sin(-20*PI/180), 0.0f, cos(-20*PI/180), 0.0f,
         0.0f,         0.0f, 0.0f,        1.0f;

    A << 1.0f, 0.0f, 0.0f, 0.0f,
         0.0f, 1.0f, 0.0f, 0.0f,
         0.0f, 0.0f, 1.0f, 2.0f,
         0.0f, 0.0f, 0.0f, 1.0f;

    cow_transformation = A*B*C;
}

void initialize_nicp_objects(){
    pointProjector.setMinDistance(point_projector_min_distance);
    pointProjector.setMaxDistance(point_projector_max_distance);
    pointProjector.setCameraMatrix(cameraMatrix);
    pointProjector.setImageSize(input_image_width, input_image_height);
    pointProjector.setTransform(Eigen::Isometry3f::Identity());
    pointProjector.scale(1.0f);        
    
    statsCalculator.setMinImageRadius(5);
    statsCalculator.setMaxImageRadius(20);
    statsCalculator.setMinPoints(10);
    statsCalculator.setCurvatureThreshold(0.1);
    statsCalculator.setWorldRadius(0.1);
    
    pointInformationMatrixCalculator.setCurvatureThreshold(0.1);
    
    normalInformationMatrixCalculator.setCurvatureThreshold(0.1);

    converter = nicp::DepthImageConverterIntegralImage(&pointProjector, &statsCalculator,
                          &pointInformationMatrixCalculator,
                          &normalInformationMatrixCalculator);
                          
    merger.setMaxPointDepth(4.5);
    merger.setNormalThreshold(0.7);
    merger.setDistanceThreshold(0.1);
    merger.setDepthImageConverter(&converter);

    // Create CorrespondenceFinder
    correspondenceFinder.setImageSize(pointProjector.imageRows(), pointProjector.imageCols());
    correspondenceFinder.setInlierDistanceThreshold(ICP_MAX_CORRESPONDECE_DISTANCE);
    correspondenceFinder.setInlierNormalAngularThreshold(0.95f);
    correspondenceFinder.setFlatCurvatureThreshold(0.02f);

    linearizer.setInlierMaxChi2(9e3);
    linearizer.setRobustKernel(true);
    linearizer.setZScaling(true);    
    linearizer.setAligner(&aligner);

    aligner.setOuterIterations(ICP_MAX_ITERATIONS);    
    aligner.setLambda(1e3);
    aligner.setProjector(&pointProjector);
    aligner.setCorrespondenceFinder(&correspondenceFinder);
    aligner.setLinearizer(&linearizer);  
}

void png_to_cloud(string depthFile, string colorFile){
    // Get clouds from depth images
    nicp::RawDepthImage rawDepth;
    nicp::DepthImage depth, scaledDepth;

    nicp::FloatImage depthImage = cv::imread(depthFile, -1);
    nicp::RGBImage colorImage = cv::imread(colorFile, 4);

    current_image = colorImage.clone();

    //cvtColor(colorImage, current_image, CV_BGR2RGB);
    cvtColor(colorImage, colorImage, CV_BGR2RGB);
    flip(colorImage, colorImage, 0);

    if(!depthImage.data || !colorImage.data) {
        std::cerr << "Error: impossible to read image file " << depthFile << std::endl;
        exit(EXIT_FAILURE);
    }

    converter.compute(currentCloud, depthImage, colorImage);
}

basic_istream<char> & read_filenames(ifstream& is, string& depthFilename, string& colorFilename){
    char buf[4096];
    is.getline(buf, 4096);
    istringstream iss(buf);
    string timestamp;

    return (iss >> timestamp >> depthFilename >> timestamp >> colorFilename);
}

void find_keypoints(vector<KeyPoint>& keypoints_previous, vector<KeyPoint>& keypoints_current){
    OrbFeatureDetector detector( MAX_FEATURES );
    detector.detect( previous_image, keypoints_previous );
    detector.detect( current_image, keypoints_current );
}

void extract_descriptor(vector<KeyPoint> keypoints_previous, vector<KeyPoint> keypoints_current, Mat& descriptors_previous, Mat& descriptors_current){
    SiftDescriptorExtractor extractor;
    extractor.compute( current_image, keypoints_previous, descriptors_previous );
    extractor.compute( previous_image, keypoints_current, descriptors_current );
}

void match_key_points(Mat descriptors_previous, Mat descriptors_current, vector<DMatch>& matches){
    BFMatcher matcher;
    matcher.match( descriptors_previous, descriptors_current, matches );
}

void remove_matches_outside_the_image(const vector<KeyPoint> keypoints_previous, const vector<KeyPoint> keypoints_current, vector<DMatch>& matches){
    nicp::IntImage index_image, index_previous_image;
    nicp::DepthImage temp;

    pointProjector.project(index_previous_image, temp, previous_cloud.points());
    pointProjector.project(index_image, temp, currentCloud.points());

    int i=0;
    while(i < matches.size()){
        int previous_keypoint_index = matches[i].queryIdx;
        int xprevious = keypoints_previous[previous_keypoint_index].pt.x;
        int yprevious = input_image_height - keypoints_previous[previous_keypoint_index].pt.y;

        int current_keypoint_index = matches[i].trainIdx;
        int xcurrent = keypoints_current[current_keypoint_index].pt.x;
        int ycurrent = input_image_height - keypoints_current[current_keypoint_index].pt.y;

        int index_current = index_image.at<int>(ycurrent,xcurrent);
        int index_previous = index_previous_image.at<int>(yprevious,xprevious);

        if(index_previous == -1 && index_current == -1){
            matches.erase(matches.begin()+i);
        }else{
            i++;
        }
    }
}

void filter_matches(vector<DMatch>& matches){
    sort(matches.begin(), matches.end());
    const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
    matches.erase(matches.begin()+numGoodMatches, matches.end());
}

void project_point(double& u, double& v, nicp::Point point){
    u = 240 + 300 * point[0] / point[2];
    v = 160 + 300 * point[1] / point[2];
}

void transform_point(nicp::Point& previous_point, nicp::Point& new_point_expected, Eigen::Matrix4f transformation){
    double x = previous_point[0];
    double y = previous_point[1];
    double z = previous_point[2];
    
    new_point_expected[0] = x * transformation(0,0) + y * transformation(0,1) + z * transformation(0,2) + transformation(0,3);
    new_point_expected[1] = x * transformation(1,0) + y * transformation(1,1) + z * transformation(1,2) + transformation(1,3);
    new_point_expected[2] = x * transformation(2,0) + y * transformation(2,1) + z * transformation(2,2) + transformation(2,3);
}

Eigen::Matrix4f perform_epnp(vector<KeyPoint> keypoints_previous, vector<KeyPoint> keypoints_current, vector<DMatch> matches){
    // Configure PnP parameters
    epnp PnP;
    PnP.set_internal_parameters(240, 160, 300, 300);
    PnP.set_maximum_number_of_correspondences(matches.size());
    PnP.reset_correspondences();

    nicp::IntImage index_previous_image, index_current_image;
    nicp::DepthImage temp;
    pointProjector.project(index_previous_image, temp, previous_cloud.points());
    pointProjector.project(index_current_image, temp, currentCloud.points());

    // Add the correspondeces
    for(int i = 0; i<matches.size(); i++){
        int previous_keypoint_index = matches[i].queryIdx;
        int xprevious = keypoints_previous[previous_keypoint_index].pt.x;
        int yprevious = input_image_height - keypoints_previous[previous_keypoint_index].pt.y;
        
        int index_previous = index_previous_image.at<int>(yprevious,xprevious);

        int current_keypoint_index = matches[i].trainIdx;
        int xcurrent = keypoints_current[current_keypoint_index].pt.x;
        int ycurrent = input_image_height - keypoints_current[current_keypoint_index].pt.y;

        int index_current = index_current_image.at<int>(ycurrent,xcurrent);

        nicp::Point point = previous_cloud.points()[index_previous];

        double x = point[0];
        double y = point[1];
        double z = point[2];

        nicp::Point new_point_expected;

        transform_point(point, new_point_expected, cow_transformation);

        double u, v, expected_u, expected_v;
        project_point(expected_u,expected_v,new_point_expected);
        u = xcurrent;
        v = ycurrent;

        double delta_u = expected_u - u;
        double delta_v = expected_v - v;

        // cout << delta_u << " x " << delta_v << endl;

        PnP.add_correspondence(x, y, z, u, v);
    }

    double R_est[3][3], t_est[3];
    double err2 = PnP.compute_pose(R_est, t_est);

    Eigen::Matrix4f result;
    result << R_est[0][0] , R_est[0][1] , R_est[0][2] , t_est[0] ,
              R_est[1][0] , R_est[1][1] , R_est[1][2] , t_est[1] ,
              R_est[2][0] , R_est[2][1] , R_est[2][2] , t_est[2] ,
              0           , 0           , 0           , 1;

    return result;
}

void draw_matches(vector<DMatch> matches, vector<DMatch>& drawn_matches, int n){
    for(int i = 0; i < n; i++){
        int index = rand() % matches.size();
        drawn_matches.push_back(matches[index]);
        matches.erase(matches.begin()+index);
    }
}

void extract_3d_points(vector<DMatch> matches, vector<KeyPoint> keypoints_previous, vector<KeyPoint> keypoints_current, vector<nicp::Point>& previous_3d_points, vector<nicp::Point>& current_3d_points){
    previous_3d_points.clear();
    current_3d_points.clear();

    nicp::IntImage index_previous_image, index_current_image;
    nicp::DepthImage temp;
    pointProjector.project(index_previous_image, temp, previous_cloud.points());
    pointProjector.project(index_current_image, temp, currentCloud.points());

    /*Mat imMatches;
    drawMatches(previous_image, keypoints_previous, current_image, keypoints_current, matches, imMatches);
    imshow("Image", imMatches);
    waitKey(0);*/

    for(int i = 0; i<matches.size(); i++){
        int previous_keypoint_index = matches[i].queryIdx;
        int xprevious = keypoints_previous[previous_keypoint_index].pt.x;
        int yprevious = input_image_height - keypoints_previous[previous_keypoint_index].pt.y;
        
        int index_previous = index_previous_image.at<int>(yprevious,xprevious);

        int current_keypoint_index = matches[i].trainIdx;
        int xcurrent = keypoints_current[current_keypoint_index].pt.x;
        int ycurrent = input_image_height - keypoints_current[current_keypoint_index].pt.y;

        int index_current = index_current_image.at<int>(ycurrent,xcurrent);

        previous_3d_points.push_back(previous_cloud.points()[index_previous]);
        current_3d_points.push_back(currentCloud.points()[index_current]);
    }
}

double calculate_inliers(Eigen::Matrix4f transformation, vector<DMatch> matches, vector<nicp::Point> previous_3d_points, vector<nicp::Point> current_3d_points, vector<DMatch>& inliers_matches, double max_distance){
    double inliers = 0;
    inliers_matches.clear();
    for(int i = 0; i < matches.size(); i++){
        nicp::Point previous_point = previous_3d_points[i];
        nicp::Point current_point = current_3d_points[i];
        nicp::Point transformed_point;

        transformed_point[0] = previous_point[0] * transformation(0,0) + previous_point[1] * transformation(0,1) + previous_point[2] * transformation(0,2) + transformation(0,3);
        transformed_point[1] = previous_point[0] * transformation(1,0) + previous_point[1] * transformation(1,1) + previous_point[2] * transformation(1,2) + transformation(1,3);
        transformed_point[2] = previous_point[0] * transformation(2,0) + previous_point[1] * transformation(2,1) + previous_point[2] * transformation(2,2) + transformation(2,3);

        double distance = sqrt(pow(transformed_point[0]-current_point[0],2) +
                               pow(transformed_point[1]-current_point[1],2) +
                               pow(transformed_point[2]-current_point[2],2));

        if(max_distance >= distance){
            inliers++;
            inliers_matches.push_back(matches[i]);
        }

    }   
    inliers = inliers / matches.size();
    return inliers;
}

Eigen::Matrix4f align_with_epnp(){
    RobustMatcher robustMatcher;

    cout << "Extracting keypoints" << endl;
    vector<KeyPoint> keypoints_previous, keypoints_current;
    find_keypoints(keypoints_previous, keypoints_current);
    //robustMatcher.computeKeyPoints( previous_image, keypoints_previous);
    //robustMatcher.computeKeyPoints( current_image, keypoints_current);

    cout << "Extracting descriptors" << endl;

    Mat descriptors_previous, descriptors_current;
    extract_descriptor(keypoints_previous, keypoints_current, descriptors_previous, descriptors_current);
    //robustMatcher.computeDescriptors( previous_image, keypoints_previous,  descriptors_previous);
    //robustMatcher.computeDescriptors( current, keypoints_current,  descriptors_current);

    cout << "Matching" << endl;

    vector<DMatch> matches;
    match_key_points(descriptors_previous, descriptors_current, matches);
    //robustMatcher.symmetryTest(matches12, matches21, matches);
    /*robustMatcher.robustMatch( current_image, matches, keypoints_current,
                       descriptors_previous);*/

    cout << matches.size() << endl;

    cout << "Filtering matches" << endl;

    remove_matches_outside_the_image(keypoints_previous, keypoints_current, matches);
    filter_matches(matches);

    cout << "extract 3d points" << endl;

    vector<nicp::Point> previous_3d_points, current_3d_points;
    extract_3d_points(matches, keypoints_previous, keypoints_current, previous_3d_points, current_3d_points);

    cout << "RANSAC" << endl;

    // Start RANSAC
    int i=0;
    double inliers = 0;
    vector<DMatch> inliers_matches;

    srand(time(NULL));// Will be used later to draw random matches
    Eigen::Matrix4f transformation;
    while(i<RANSAC_MAX_ITERATIONS && inliers < RANSAC_MIN_INLIERS){
        vector<DMatch> drawn_matches;
        draw_matches(matches, drawn_matches, 10);
        transformation = perform_epnp(keypoints_previous, keypoints_current, drawn_matches);
        inliers = calculate_inliers(transformation, matches, previous_3d_points, current_3d_points, inliers_matches, RANSAC_INLIER_DISTANCE);
        i++;
    }

    if(inliers>RANSAC_MIN_INLIERS){
        return perform_epnp(keypoints_previous, keypoints_current, inliers_matches);
    }
    else
        return Eigen::Matrix4f::Identity();
}

double average_point_error(Eigen::Matrix4f T0, Eigen::Matrix4f T1, nicp::Cloud original_cloud){
    nicp::Cloud cloud0, cloud1;
    
    // Copy both clouds
    cloud0.add(original_cloud);
    cloud1.add(original_cloud);

    Eigen::Isometry3f T;
    T.matrix() = T0;
    cloud0.transformInPlace(T);

    T.matrix() = T1;
    cloud1.transformInPlace(T);

    double total_error = 0;
    
    for(size_t i=0; i<original_cloud.points().size(); i++){
        nicp::Point point0 = cloud0.points()[i];
        nicp::Point point1 = cloud1.points()[i];
        double euclidian_distance = sqrt(
            pow(point0[0]-point1[0], 2) +
            pow(point0[1]-point1[1], 2) +
            pow(point0[2]-point1[2], 2));
        total_error += euclidian_distance;
    }
    return (total_error / original_cloud.points().size());
}

Eigen::Matrix4f align_with_nicp(Eigen::Matrix4f initial_alignment){
    Eigen::Isometry3f sensorOffset = Eigen::Isometry3f::Identity();
    Eigen::Isometry3f initialGuess;
    initialGuess.matrix() = initial_alignment;

    // Perform the registration
    aligner.setReferenceCloud(&currentCloud);
    aligner.setCurrentCloud(&nicp_complete_cloud);
    aligner.setInitialGuess(initialGuess);
    aligner.setSensorOffset(sensorOffset);
    cout << "Starting ICP" << endl;
    aligner.align();
    cout << "End of the ICP" << endl;

    Eigen::Isometry3f T = aligner.T();
    cout << "ICP transformation: " << T.matrix() << endl;
    return T.matrix();  
}

double perform_alignment(){
    Eigen::Matrix4f transformation = align_with_epnp();
    //Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
    cout << "Initial alignment:" << endl << transformation << endl;

    transformation = align_with_nicp(transformation);
    Eigen::Isometry3f T;
    T.matrix() = transformation;
    nicp_complete_cloud.transformInPlace(T);
    cout << "Expected transformation:" << endl << cow_transformation << endl;

    nicp_complete_cloud.add(currentCloud);
    merger.merge(&nicp_complete_cloud);

    return average_point_error(transformation.inverse(), cow_transformation.inverse(), currentCloud);

}

int main(int argc, char *argv[])
{
    //Reading arguments
    if(argc < 2) {
        std::cout << "Usage: 3d reconstruction <associations.txt>" << std::endl
                  << "\tassociations.txt\t-->\tfiel containing a set of depth images associations for alignment in the format: " << std::endl;
        return 0;
    }
    string associationsFile = string(argv[1]);

    // Order of initialization is important
    initialize_matrixes();
    initialize_nicp_objects();

    ifstream is(associationsFile.c_str());
    if(!is) {
        std::cerr << "[ERROR]: impossible to open depth images associations file " << associationsFile << std::endl;
        return -1;
    }

    bool first_frame=true;
    int frame = 0;

    double error = 0;

    while(is.good()) {
        cout << frame++ << endl;

        string depthFilename, colorFilename;
        if(!read_filenames(is, depthFilename, colorFilename))
            continue;
        
        png_to_cloud(depthFilename, colorFilename);

        //Performs alignment and merge
        if(first_frame){
            nicp_complete_cloud.add(currentCloud);
            first_frame=false;
        }else{
            double frame_error = perform_alignment();
            cout << "Frame error: " << frame_error << endl;
            error += frame_error;
        }

        previous_image = current_image.clone();
        previous_cloud = currentCloud;
        cout << "Partial error: " << error / frame << endl;
        cout << endl << "================================================================================" << endl << endl;
    }
    
    return 0;
}
