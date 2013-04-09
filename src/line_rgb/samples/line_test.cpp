#include <iterator>
#include <set>
#include <cstdio>
#include <sstream>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "objdetect_line_rgb.hpp"

#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>

using namespace std;
using namespace cv;

static void help_f() {
    printf(
            "\n\n"
                    " Usage: line_test [options] [actions] [modality] \n\n"
                    " This test can run a Line2D or a LineRGB detector on a sequence of images to detect,\n"
                    " a trained objects in the scene. The templates of the object can be provided using the\n"
                    " --train option or loading a .yml file by the --load option. The .yml file can be generated \n"
                    " using together the --train and --save options, or pressing 'w' during test execution; in \n"
                    " both cases the file will be stored in the same folder of this executable. \n\n"
                    " usage example 1: line_test -t [templates_folder] -v [video_folder] --save --train --test --linergb\n"
                    " usage example 2: line_test -v [video_folder] --load --test --linergb\n"
                    " usage example 3: line_test -t [templates_folder] --save --line2d\n\n"
                    "Options:\n"
                    "\t -h  	   -- This help page\n"
                    "\t -t  	   -- provides the path to the templates folder\n"
                    "\t -v  	   -- provides the path to the video folder\n"
                    "\t --train   -- trains a detector with the chosen modality using the templates folder specified by -t\n"
                    "\t --test    -- search for an object trained in a video specified by -v, using the chosen modality\n"
                    "\t --load    -- load a detector of the chosen modality using a pre-saved .yml\n"
                    "\t --save    -- write a detector of the chosen modality using the templates just trained\n"
                    "\t --linergb -- specifies to use the LineRGB modality\n"
                    "\t --line2d  -- specifies to use the original Line2D modality\n\n"
                    "Keys on test running:\n"
                    "\t w   -- write the detector on a file\n"
                    "\t p   -- pause the test\n"
                    "\t q   -- Quit\n\n");
}

// Adapted from cv_timer in cv_utilities
class Timer {
public:
    Timer() :
            start_(0), time_(0) {
    }

    void start() {
        start_ = cv::getTickCount();
    }

    void stop() {
        CV_Assert(start_ != 0);
        int64 end = cv::getTickCount();
        time_ += end - start_;
        start_ = 0;
    }

    double time() {
        double ret = time_ / cv::getTickFrequency();
        time_ = 0;
        return ret;
    }

private:
    int64 start_, time_;
};

cv::Ptr<cv::line_rgb::Detector> readLineRGB(const std::string& filename);

void writeLineRGB(const cv::Ptr<cv::line_rgb::Detector>& detector,
        const std::string& filename);

static cv::Ptr<cv::linemod::Detector> readLine2D(const std::string& filename);

static void writeLine2D(const cv::Ptr<cv::linemod::Detector>& detector,
        const std::string& filename);

void drawResponseLineRGB(const std::vector<cv::line_rgb::Template>& templates,
        int num_modalities, cv::Mat& dst, cv::Point offset, int T,
        short rejected, string class_id);

void drawResponseLine2D(const std::vector<cv::linemod::Template>& templates,
        int num_modalities, cv::Mat& dst, cv::Point offset, int T,
        short rejected, string class_id);

Mat rotateImage(const Mat& source, double angle);

void readDirectory(const char* dirname, vector<string>& list_files);

string getColorFromMask(string mask);

string getDepthFromMask(string mask);

void getMasksFromListFile(vector<string>& list_files);

int getMaskNumber(string mask);

string charToString(char* c);

int stringToInt(string s);

void splitByUnderscore(string src, string& word1, string& word2);
void splitBySlash(string src, string& word1, string& word2);

int main(int argc, char * argv[]) {
    if (argc <= 2) {
        help_f();
        return 0;
    }

    bool train = false;
    bool templates_given = false;
    bool video_given = false;
    bool test = false;
    bool load = false;
    bool save = false;
    bool line2d = false;
    bool linergb = false;
    bool help = false;

    int num_modalities = 1;

    string path_name = "";
    string path_templates = "";
    string path_video = "";

    for (int h = 1; h <= (argc - 1); h++) {
        if (strcmp("-t", argv[h]) == 0) {
            path_templates = charToString(argv[h + 1]);
            h++;
            templates_given = true;
        }
        if (strcmp("-v", argv[h]) == 0) {
            path_video = charToString(argv[h + 1]);
            h++;
            video_given = true;
        }

        if (strcmp("--load", argv[h]) == 0) {
            load = true;
            printf("Load templates\n");
        }

        if (strcmp("--train", argv[h]) == 0) {
            train = true;
            printf("TRAIN ENABLED\n");
        }

        if (strcmp("--save", argv[h]) == 0) {
            save = true;
        }

        if (strcmp("--test", argv[h]) == 0) {
            test = true;
        }
        if (strcmp("--line2d", argv[h]) == 0) {
            line2d = true;
        }
        if (strcmp("--linergb", argv[h]) == 0) {
            linergb = true;
        }
        if (strcmp("-h", argv[h]) == 0) {
            help = true;
        }

    }

    if (help == true) {
        help_f();
        return 0;
    }

    if (train == false && test == false) {
        printf("Provide --test and/or --train option\n");
        printf("use -h option to show help\n");
        return 0;
    }
    if (line2d == true && linergb == true) {
        printf(
                "LineRGB and Line2D are exclusive. Please choose only one modality\n");
        printf("use -h option to show help\n");
        return 0;
    } else if (line2d == true) {
        printf("Test will be executed with Line2D\n");
    } else if (linergb == true) {
        printf("Test will be executed with LineRGB\n");
    } else if (line2d == false && linergb == false) {
        printf("Please specify one modality: \"--linergb\" or \"--line2d\"\n");
        printf("use -h option to show help\n");
        return 0;
    }
    if (train == true && load == true) {
        printf(
                "\"--load\" option is enabled. Option \"--train\" will be ignored\n");
        train = false;
    }
    if (train == true && templates_given == false) {
        printf(
                "A templates folder is needed to train, please use the \"-t\" option\n");
        printf("use -h option to show help\n");
        return 0;
    }
    if (train == false && save == true) {
        printf("Could not save templates if \"--train\" is not enabled:\n");
    }
    if (test == true && video_given == false) {
        printf(
                "A video folder is needed to test, please use the \"-v\" option\n");
        printf("use -h option to show help\n");
        return 0;
    }
    if (test == true) {
        if (train == true || load == true) {
            printf("TEST ENABLED\n");
        } else {
            printf(
                    "Could not run test without templates. \"--train\" or \"--load\" must be enabled\n");
            printf("use -h option to show help\n");
            return 0;
        }
    }

    string templates_folder = "";
    path_name = "";
    string object_name = "";
    string instance = "";

    splitBySlash(path_templates, templates_folder, path_name);
    splitByUnderscore(path_name, object_name, instance);

    string class_id = object_name + "_" + instance;
    string object_folder = object_name + "/" + class_id + "/";

    string video_folder = "";
    string video_name = "";
    splitBySlash(path_video, video_folder, video_name);

    /////-TRAIN-/////
    Timer total_train_timer;
    total_train_timer.start();

    cv::Ptr < cv::line_rgb::Detector > detector_rgb;
    cv::Ptr < cv::linemod::Detector > detector_line2d;
    if (linergb == true)
        detector_rgb = line_rgb::getDefaultLINERGB();
    if (line2d == true)
        detector_line2d = linemod::getDefaultLINE();

    if (train == true) {
        vector < string > list_files;

        string cartella = templates_folder;
        readDirectory(cartella.c_str(), list_files);
        //retrieve all mask names from object folder
        getMasksFromListFile (list_files);
        int size_files = list_files.size();

        // Extract templates

        string filename;
        if (linergb == true)
            filename = "./line_rgb_" + class_id + ".yml";
        if (line2d == true)
            filename = "./line_2d_" + class_id + ".yml";

        for (int i = 0; i < size_files; i++) {
            stringstream out;
            out << i;

            string current_mask = list_files.at(i);

            int index_temp = getMaskNumber(current_mask);

            if (index_temp % 5 == 0) {

                string template_image = templates_folder
                        + getColorFromMask(current_mask);
                string mask_image = templates_folder + current_mask;
                string depth_image = templates_folder
                        + getDepthFromMask(current_mask);

                double resizes[6] = { 0.7, 0.8, 0.9, 1, 1.2, 1.4 };
                double rotations[3] = { -22.5, 1.0, 22.5 };

                cv::Mat mask;
                mask = cv::imread(mask_image, 0);

                if (mask.data != NULL) {
                    cv::Mat single_source;
                    single_source = cv::imread(template_image, 1);
                    cv::Mat single_sourceDepth;
                    single_sourceDepth = cv::imread(depth_image, 1);

                    for (int iter = 0; iter < 6; iter++) {
                        double resize_factor = resizes[iter];
                        cv::Mat single_source_dst;
                        cv::Mat mask_dst;
                        if (resize_factor == 1.0) {
                            single_source_dst = single_source;
                            mask_dst = mask;
                        } else {
                            if (resize_factor < 1.0) {
                                cv::resize(single_source, single_source_dst,
                                        Size(), resize_factor, resize_factor); //, CV_INTER_AREA);
                                cv::resize(mask, mask_dst, Size(),
                                        resize_factor, resize_factor); //, CV_INTER_AREA);

                            } else {
                                cv::resize(single_source, single_source_dst,
                                        Size(), resize_factor, resize_factor,
                                        CV_INTER_CUBIC);
                                cv::resize(mask, mask_dst, Size(),
                                        resize_factor, resize_factor,
                                        CV_INTER_CUBIC);
                            }
                        }

                        for (int iterRot = 0; iterRot < 3; iterRot++) {
                            double rotation_factor = rotations[iterRot];
                            cv::Mat single_source_final;
                            cv::Mat mask_final;
                            if (rotation_factor == 1.0) {
                                single_source_final = single_source_dst.clone();
                                mask_final = mask_dst;
                            } else {
                                single_source_final = rotateImage(
                                        single_source_dst, rotation_factor);
                                mask_final = rotateImage(mask_dst,
                                        rotation_factor);
                            }

                            cv::Rect bb;
                            std::vector < cv::Mat > sourcesTemplate;
                            sourcesTemplate.push_back(single_source_final);
                            if (num_modalities == 2)
                                sourcesTemplate.push_back(single_sourceDepth);

                            Timer single_train_timer;
                            single_train_timer.start();
                            int template_id;

                            if (linergb == true)
                                template_id = detector_rgb->addTemplate(
                                        sourcesTemplate, class_id, mask_final,
                                        &bb);
                            if (line2d == true)
                                template_id = detector_line2d->addTemplate(
                                        sourcesTemplate, class_id, mask_final,
                                        &bb);

                            single_train_timer.stop();
                            printf("Train single: %.2fs\n",
                                    single_train_timer.time());
                            printf("\n.....templating...");
                            cout << class_id << endl;
                            if (template_id != -1) {
                                cout << "*** Added template (id " << template_id
                                        << ") for new object class " << class_id
                                        << " - path:" << template_image << "***"
                                        << endl;
                            }

                        }

                    }

                } else {
                    cout << mask_image << " not found" << std::endl;
                }
            }
        }
        if (save == true) {
            if (linergb == true)
                writeLineRGB(detector_rgb, filename);
            if (line2d == true)
                writeLine2D(detector_line2d, filename);
            cout << endl << filename << " saved" << endl;
        }
        total_train_timer.stop();
        cout << "Whole train time: " << total_train_timer.time() << endl;
    }

    if (test == true) {
        ///////////////
        /////-TEST-/////

        Timer global_test_timer;
        global_test_timer.start();

        vector < string > list_files;
        readDirectory(video_folder.c_str(), list_files);
        int framesNumber = list_files.size() / 2;

        string filename;
        if (linergb == true)
            filename = "./line_rgb_" + class_id + ".yml";
        if (line2d == true)
            filename = "./line_2d_" + class_id + ".yml";

        cv::Mat color, depth;
        int num_classes = 1;

        if (load == true) {
            if (linergb == true)
                detector_rgb = readLineRGB(filename);
            if (line2d == true)
                detector_line2d = readLine2D(filename);

            cout << endl << filename << " saved" << endl;
        }

        std::vector < std::string > ids;
        if (linergb == true) {
            ids = detector_rgb->classIds();
            num_classes = detector_rgb->numClasses();
            printf("\nLoaded %s with %d classes and %d templates\n", argv[1],
                    num_classes, detector_rgb->numTemplates());
        }
        if (line2d == true) {
            ids = detector_line2d->classIds();
            num_classes = detector_line2d->numClasses();
            printf("\nLoaded %s with %d classes and %d templates\n", argv[1],
                    num_classes, detector_line2d->numTemplates());
        }

        if (!ids.empty()) {
            printf("Class ids:\n");
            std::copy(ids.begin(), ids.end(),
                    std::ostream_iterator < std::string > (cout, "\n"));
        }

        int current = 1;

        /////////////////
        /////-MATCH-/////

        while (current <= framesNumber) {
            cout << endl << "FRAME " << current << endl;
            stringstream out;
            out << current;

            string roi_image = video_folder + video_name + "_" + out.str()
                    + ".png";
            string roi_depth_image = video_folder + video_name + "_" + out.str()
                    + "_depth" + ".png";

            color = cv::imread(roi_image, 1);

            if (color.data != NULL) {
                vector < cv::Mat > sources;
                sources.push_back(color);

                if (num_modalities == 2) {
                    depth = cv::imread(roi_depth_image, 1);
                    sources.push_back(depth);
                }

                cv::Mat display;
                display = color.clone();

                if (linergb == true) {
                    // Perform matching
                    vector < cv::line_rgb::Match > matches;
                    vector < std::string > class_ids;
                    vector < cv::Mat > quantized_images;
                    Timer match_timer;
                    match_timer.start();
                    detector_rgb->match(sources, 80, matches, class_ids,
                            quantized_images);
                    match_timer.stop();
                    printf("Matching: %.2fs\n", match_timer.time());
                    int classes_visited = 0;
                    std::set < std::string > visited;

                    for (int i = 0;
                            i < (int) matches.size()
                                    && (classes_visited < num_classes); ++i) {
                        cv::line_rgb::Match m = matches[i];

                        if (visited.insert(m.class_id).second) {
                            ++classes_visited;

                            printf(
                                    "Similarity combined: %5.1f%%; Similarity_2d: %5.1f%%; Similarity_rgb: %5.1f%%; x: %3d; y: %3d; class: %s; template: %3d",
                                    m.sim_combined, m.similarity,
                                    m.similarity_rgb, m.x, m.y,
                                    m.class_id.c_str(), m.template_id);

                            // Draw matching template
                            const std::vector<cv::line_rgb::Template>& templates =
                                    detector_rgb->getTemplates(m.class_id,
                                            m.template_id);
                            drawResponseLineRGB(templates, num_modalities,
                                    display, cv::Point(m.x, m.y),
                                    detector_rgb->getT(0), m.rejected,
                                    m.class_id.c_str());

                        }

                    }
                    if (matches.empty())
                        printf("No matches found...\n");

                    printf("Matching: %.2fs\n", match_timer.time());
                    printf(
                            "\n------------------------------------------------------------\n");

                    cv::imshow("LineRGB", display);

                } else if (line2d == true) {
                    vector < cv::linemod::Match > matches;
                    vector < std::string > class_ids;
                    vector < cv::Mat > quantized_images;
                    Timer match_timer;
                    match_timer.start();
                    detector_line2d->match(sources, 80, matches, class_ids,
                            quantized_images);
                    match_timer.stop();
                    printf("Matching: %.2fs\n", match_timer.time());
                    int classes_visited = 0;
                    std::set < std::string > visited;

                    for (int i = 0;
                            i < (int) matches.size()
                                    && (classes_visited < num_classes); ++i) {
                        cv::linemod::Match m = matches[i];

                        if (visited.insert(m.class_id).second) {
                            ++classes_visited;

                            printf(
                                    "Similarity_2d: %5.1f%%; x: %3d; y: %3d; class: %s; template: %3d",
                                    m.similarity, m.x, m.y, m.class_id.c_str(),
                                    m.template_id);

                            // Draw matching template
                            const std::vector<cv::linemod::Template>& templates =
                                    detector_line2d->getTemplates(m.class_id,
                                            m.template_id);
                            drawResponseLine2D(templates, num_modalities,
                                    display, cv::Point(m.x, m.y),
                                    detector_line2d->getT(0), -1,
                                    m.class_id.c_str());

                        }

                    }
                    if (matches.empty())
                        printf("No matches found...\n");

                    printf(
                            "\n------------------------------------------------------------\n");

                    cv::imshow("Line2D", display);

                }

                if (current == 1)
                    waitKey(0);

                char key = (char) cvWaitKey(10);
                switch (key) {
                case 'w':
                    // write model to disk
                    if (linergb == true)
                        writeLineRGB(detector_rgb, filename);
                    if (line2d == true)
                        writeLine2D(detector_line2d, filename);
                    printf("Wrote detector and templates to %s\n",
                            filename.c_str());
                    break;
                case 'p':
                    // pause
                    waitKey();
                    break;
                case 'q':
                    return 0;
                }

            } else {
                cout << roi_image << "not found" << std::endl;
            }
            current++;
        }

        global_test_timer.stop();
        cout << "Global test time: " << global_test_timer.time() << endl;
    }		    //test

    return 0;
}

void drawResponseLineRGB(const std::vector<cv::line_rgb::Template>& templates,
        int num_modalities, cv::Mat& dst, cv::Point offset, int T,
        short rejected, string class_id) {

    cv::Scalar color;
    for (int m = 0; m < num_modalities; ++m) {

        for (int i = 0; i < (int) templates[m].features.size(); ++i) {
            cv::line_rgb::Feature f = templates[m].features[i];
            cv::Point pt(f.x + offset.x, f.y + offset.y);
            switch (f.rgb_label) {
            case 0:
                color = CV_RGB(255, 0, 0);
                break;
            case 1:
                color = CV_RGB(0, 255, 0);
                break;
            case 2:
                color = CV_RGB(0, 0, 255);
                break;
            case 3:
                color = CV_RGB(255, 255, 0);
                break;
            case 4:
                color = CV_RGB(255, 0, 255);
                break;
            case 5:
                color = CV_RGB(0, 255, 255);
                break;
            case 6:
                color = CV_RGB(255, 255, 255);
                break;
            case 7:
                color = CV_RGB(0, 0, 0);
                break;
            }

            cv::circle(dst, pt, T / 2, color);
        }

    }
}

void drawResponseLine2D(const std::vector<cv::linemod::Template>& templates,
        int num_modalities, cv::Mat& dst, cv::Point offset, int T,
        short rejected, string class_id) {

    cv::Scalar color;
    for (int m = 0; m < num_modalities; ++m) {

        for (int i = 0; i < (int) templates[m].features.size(); ++i) {
            cv::linemod::Feature f = templates[m].features[i];
            cv::Point pt(f.x + offset.x, f.y + offset.y);

            color = CV_RGB(0, 0, 255);

            cv::circle(dst, pt, T / 2, color);
        }

    }
}

// Functions to store detector and templates in single XML/YAML file
cv::Ptr<cv::line_rgb::Detector> readLineRGB(const std::string& filename) {
    cv::Ptr < cv::line_rgb::Detector > detector = new cv::line_rgb::Detector;
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    detector->read(fs.root());

    cv::FileNode fn = fs["classes"];
    for (cv::FileNodeIterator i = fn.begin(), iend = fn.end(); i != iend; ++i)
        detector->readClass(*i);

    return detector;
}

void writeLineRGB(const cv::Ptr<cv::line_rgb::Detector>& detector,
        const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    detector->write(fs);

    std::vector < std::string > ids = detector->classIds();
    fs << "classes" << "[";
    for (int i = 0; i < (int) ids.size(); ++i) {
        fs << "{";
        detector->writeClass(ids[i], fs);
        fs << "}"; // current class
    }
    fs << "]"; // classes
    //fs.releaseAndGetString();
}

// Functions to store detector and templates in single XML/YAML file
static cv::Ptr<cv::linemod::Detector> readLine2D(const std::string& filename) {
    cv::Ptr < cv::linemod::Detector > detector = new cv::linemod::Detector;
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    detector->read(fs.root());

    cv::FileNode fn = fs["classes"];
    for (cv::FileNodeIterator i = fn.begin(), iend = fn.end(); i != iend; ++i)
        detector->readClass(*i);

    return detector;
}

static void writeLine2D(const cv::Ptr<cv::linemod::Detector>& detector,
        const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    detector->write(fs);

    std::vector < cv::String > ids = detector->classIds();
    fs << "classes" << "[";
    for (int i = 0; i < (int) ids.size(); ++i) {
        fs << "{";
        detector->writeClass(ids[i], fs);
        fs << "}"; // current class
    }
    fs << "]"; // classes
}

Mat rotateImage(const Mat& source, double angle) {
    Point2f src_center(source.cols / 2.0F, source.rows / 2.0F);
    Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
    Mat dst;
    warpAffine(source, dst, rot_mat, source.size(), INTER_CUBIC);
    return dst;
}

void readDirectory(const char* dirname, vector<string>& list_files) {

    DIR *dir;
    struct dirent *ent;
    dir = opendir(dirname);
    if (dir != NULL) {

        /* print all the files and directories within directory */
        while ((ent = readdir(dir)) != NULL) {
            //printf ("%s\n", ent->d_name);
            list_files.push_back(charToString(ent->d_name));
        }
        //sort (list_files.begin(), list_files.end(), stringCompare);
        closedir(dir);
    } else {
        /* could not open directory */
        fprintf(stderr, "ERROR: Could not open directory %s\n", dirname);
    }

}

string getColorFromMask(string mask) {
    string color;
    size_t lenght = mask.size() - 9; //9 = _mask.png
    color = mask.substr(0, lenght);
    color = color + ".png";

    return color;
}

string getDepthFromMask(string mask) {
    string depth;
    size_t lenght = mask.size() - 9; //9 = _mask.png
    depth = mask.substr(0, lenght);
    depth = depth + "_depth.png";

    return depth;
}

void getMasksFromListFile(vector<string>& list_files) {
    vector < string > masks;
    for (int i = 0; i < list_files.size(); i++) {
        string temp_string = list_files.at(i);
        if (temp_string.find("_mask.png") != string::npos) {
            masks.push_back(temp_string);
            getColorFromMask(temp_string);
            getDepthFromMask(temp_string);

        }
    }
    list_files.clear();
    list_files = masks;
}

string last_occur(const char* str1, const char* str2) {
    char* strp;
    int len1, len2;

    len2 = strlen(str2);
    if (len2 == 0)
        return (char*) str1;

    len1 = strlen(str1);
    if (len1 - len2 <= 0)
        return 0;

    strp = (char*) (str1 + len1 - len2);
    while (strp != str1) {
        if (*strp == *str2) {
            if (strncmp(strp, str2, len2) == 0)
                return charToString(strp);
        }
        strp--;
    }
    return 0;
}

int getMaskNumber(string mask) {
    string tmp = last_occur(mask.c_str(), "_");

    string cutted = mask.substr(0, mask.size() - tmp.size());
    string final = last_occur(cutted.c_str(), "_");
    final = final.substr(1, final.size());

    return stringToInt(final);
}

int stringToInt(string s) {
    stringstream ss;
    int number;
    ss << s;
    ss >> number;

    return number;
}

string charToString(char* c) {
    stringstream ss;
    string s;
    ss << c;
    ss >> s;

    return s;
}

void splitByUnderscore(string src, string& word1, string& word2) {
    string tmp = last_occur(src.c_str(), "_");

    word1 = src.substr(0, src.size() - tmp.size());
    word2 = tmp.substr(1, 1);
}

void splitBySlash(string src, string& word1, string& word2) {
    string tmp = last_occur(src.c_str(), "/");
    if (tmp == "/") {
        src = src.substr(0, src.size() - 1);
        tmp = last_occur(src.c_str(), "/");
        src = src + "/";
    } else {
        src = src + "/";
    }

    word1 = src;
    word2 = tmp.substr(1, tmp.size() - 1);
}

