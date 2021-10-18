#include <thread>
#include <iostream>
#include <queue> 
#include <fstream>
#include <iostream>
#include <experimental/filesystem>
#include <string>
#include <list>
#include <chrono>
#include <functional>
#include <random>
#include <memory>
#include <vector>
#include <algorithm>
#include <iterator>

#include <gflags/gflags.h>

#include <inference_engine.hpp>
#include <ngraph/ngraph.hpp>

#include <opencv2/opencv.hpp>
#include <samples/ocv_common.hpp>


#include <stdlib.h>     /* getenv */

using namespace cv;
using namespace InferenceEngine::details;
using namespace InferenceEngine;

namespace fs = std::experimental::filesystem;
typedef std::chrono::milliseconds ms;

static const char help_message[]   = "Print a usage message";
static const char input_message[]  = "Required. Path to folder of images to input";
static const char model_message[]  = "Required. Path to model file.";
static const char devices_message[]  = "Required. Number devices.";
static const char labels_message[]  = "Required. Labels file.";
static const char outfile_message[] = "Required. Output file.";
static const char input_resizable_message[] = "Optional. Enable resizable input with support of ROI crop and auto resize.";


DEFINE_bool(h, false, help_message);
DEFINE_string(imgpath, "", input_message);
DEFINE_string(model, "", model_message);
DEFINE_int32(num_device, 0, devices_message);
DEFINE_string(labelpath, "", labels_message);
DEFINE_string(outfilepath, "./tmp.txt", outfile_message);
DEFINE_bool(auto_resize, false, input_resizable_message);


std::vector<std::string> labeldata;

void createLabels(std::string path){
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)){
        std::istringstream iss(line);
        std::string label;
        if (!(iss >> label)) { break; } // error

        labeldata.push_back(label);
    }
}


void FrameToBlob(const cv::Mat &frame, InferRequest::Ptr &inferRequest, const std::string &inputName) {
    if (FLAGS_auto_resize) {
        /* Just set input blob containing read image. Resize and layout conversion will be done automatically */
        inferRequest->SetBlob(inputName, wrapMat2Blob(frame));
    } else {
        /* Resize and copy data from the image to the input blob */
        Blob::Ptr frameBlob = inferRequest->GetBlob(inputName);
        matU8ToBlob<uint8_t>(frame, frameBlob);
    }
}

static int EntryIndex(int side, int lcoords, int lclasses, int location, int entry) {
    int n = location / (side * side);
    int loc = location % (side * side);
    return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
}

struct DetectionObject {
    int xmin, ymin, xmax, ymax, class_id;
    float confidence;

    DetectionObject(double x, double y, double h, double w, int class_id, float confidence, float h_scale, float w_scale) {
        this->xmin = static_cast<int>((x - w / 2) * w_scale);
        this->ymin = static_cast<int>((y - h / 2) * h_scale);
        this->xmax = static_cast<int>(this->xmin + w * w_scale);
        this->ymax = static_cast<int>(this->ymin + h * h_scale);
        this->class_id = class_id;
        this->confidence = confidence;
    }

    bool operator <(const DetectionObject &s2) const {
        return this->confidence < s2.confidence;
    }
    bool operator >(const DetectionObject &s2) const {
        return this->confidence > s2.confidence;
    }
};

class YoloParams {
    template <typename T>
    void computeAnchors(const std::vector<T> & mask) {
        std::vector<float> maskedAnchors(num * 2);
        for (int i = 0; i < num; ++i) {
            maskedAnchors[i * 2] = anchors[mask[i] * 2];
            maskedAnchors[i * 2 + 1] = anchors[mask[i] * 2 + 1];
        }
        anchors = maskedAnchors;
    }

public:
    int num = 0, classes = 0, coords = 0;
    std::vector<float> anchors = {10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0,
                                  156.0, 198.0, 373.0, 326.0};

    YoloParams() {}

    YoloParams(const std::shared_ptr<ngraph::op::RegionYolo> regionYolo) {
        coords = regionYolo->get_num_coords();
        classes = regionYolo->get_num_classes();
        anchors = regionYolo->get_anchors();
        auto mask = regionYolo->get_mask();
        num = mask.size();

        computeAnchors(mask);
    }
};

void ParseYOLOV3Output(const YoloParams &params, const std::string & output_name,
                       const Blob::Ptr &blob, const unsigned long resized_im_h,
                       const unsigned long resized_im_w, const unsigned long original_im_h,
                       const unsigned long original_im_w, int imgIdx,
                       std::vector<std::pair<int, DetectionObject> >* allObjects) {

    const int out_blob_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);
    const int out_blob_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);
    if (out_blob_h != out_blob_w)
        throw std::runtime_error("Invalid size of output " + output_name +
        " It should be in NCHW layout and H should be equal to W. Current H = " + std::to_string(out_blob_h) +
        ", current W = " + std::to_string(out_blob_h));

    auto side = out_blob_h;
    auto side_square = side * side;
    LockedMemory<const void> blobMapped = as<MemoryBlob>(blob)->rmap();
    const float *output_blob = blobMapped.as<float *>();
    // --------------------------- Parsing YOLO Region output -------------------------------------
    for (int i = 0; i < side_square; ++i) {
        int row = i / side;
        int col = i % side;
        for (int n = 0; n < params.num; ++n) {
            int obj_index = EntryIndex(side, params.coords, params.classes, n * side * side + i, params.coords);
            int box_index = EntryIndex(side, params.coords, params.classes, n * side * side + i, 0);
            float scale = output_blob[obj_index];
            if (scale < 0.5) {
                continue;
            }
            double x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w;
            double y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h;
            double height = std::exp(output_blob[box_index + 3 * side_square]) * params.anchors[2 * n + 1];
            double width = std::exp(output_blob[box_index + 2 * side_square]) * params.anchors[2 * n];
            for (int j = 0; j < params.classes; ++j) {
                int class_index = EntryIndex(side, params.coords, params.classes, n * side_square + i, params.coords + 1 + j);
                float prob = scale * output_blob[class_index];
                DetectionObject obj(x, y, height, width, j, prob,
                        static_cast<float>(original_im_h) / static_cast<float>(resized_im_h),
                        static_cast<float>(original_im_w) / static_cast<float>(resized_im_w));
                if (prob >= 0.01) {
                    allObjects->push_back({imgIdx,obj});
                }
            }
        }
    }
}

class NcsClassifier{
	public:
		int id;
		int current_request_id;
		int next_request_id;
		std::queue<std::pair<int,Mat> > queue;
        std::vector<std::pair<int, DetectionObject> >* results;
        std::map<std::string, YoloParams> yoloParams;
        long resized_im_h;
        long resized_im_w;
		std::string inputName;
        std::string outputName;
        OutputsDataMap outputInfo;
		ExecutableNetwork executable_network;
		InferRequest::Ptr async_infer_request_next;
		InferRequest::Ptr async_infer_request_curr;
		NcsClassifier(int id, std::queue<std::pair<int, Mat> > queue, std::string model_xml);
		void load_model(std::string model_xml);
		void predict_async(int imgIdx, Mat image);

        // Performance Counters:
        int * num_frame_processed;
};

NcsClassifier::NcsClassifier(int id, std::queue<std::pair<int, Mat> > queue, std::string model_xml){
	    this->id = id;
        this->current_request_id = 0;
        this->next_request_id = 1;
        this->queue = queue;
        this->results = new std::vector<std::pair<int, DetectionObject> >();
        load_model(model_xml);
		std::cout << "NcsClassifier completed" << std::endl;

        this->num_frame_processed = new int;
        *this->num_frame_processed = 0;
}

void NcsClassifier::load_model(std::string model_xml){
	// ---------------------1. Load inference engine ---------------------------------------------------------------------
        std::cout << "Loading Inference Engine" << std::endl;
        Core ie;
        std::cout << "Device info" << std::endl;
        //std::cout << ie.GetVersions("MYRIAD")        
	// ---------------------2. Read IR Generated by ModelOptimizer (.xml and .bin files) -----------------------
        std::cout << "Loading network files" << std::endl;
        auto cnnNetwork = ie.ReadNetwork(model_xml);
        /** Set batch size to 1 **/
        std::cout << "Batch size is forced to  1." << std::endl;
        cnnNetwork.setBatchSize(1);
        /** Read labels (if any)**/
        fs::path p(model_xml);
        std::cout << p.stem().string() << std::endl;
        std::string labelFileName = p.stem().string() + ".labels";
        std::vector<std::string> labels;
        std::ifstream inputFile(labelFileName);
        std::copy(std::istream_iterator<std::string>(inputFile),
                  std::istream_iterator<std::string>(),
                  std::back_inserter(labels));
		// -----------------------------------------------------------------------------------------------------

        /** SSD-based network should have one input and one output **/
        // --------------------------- 3. Configure input & output ---------------------------------------------
        // --------------------------- Prepare input blobs -----------------------------------------------------
        std::cout << "Checking that the inputs are as the demo expects" << std::endl;
        InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("This demo accepts networks having only one input");
        }
        InputInfo::Ptr& input = inputInfo.begin()->second;
        this->inputName = inputInfo.begin()->first;
        input->setPrecision(Precision::U8);
        input->getInputData()->setLayout(Layout::NCHW);
        const TensorDesc& inputDesc = inputInfo.begin()->second.get()->getTensorDesc();
        this->resized_im_h = getTensorHeight(inputDesc);
        this->resized_im_w = getTensorWidth(inputDesc);

        // --------------------------- Prepare output blobs -----------------------------------------------------
        std::cout << "Checking that the outputs are as the demo expects" << std::endl;
        this->outputInfo = OutputsDataMap(cnnNetwork.getOutputsInfo());
        for (auto &output : outputInfo) {
            output.second->setPrecision(Precision::FP32);
            output.second->setLayout(Layout::NCHW);
        }

        //DataPtr& output = outputInfo.begin()->second;
        this->outputName = outputInfo.begin()->first;
        //int num_classes = 0;
        if (auto ngraphFunction = cnnNetwork.getFunction()) {
            for (const auto op : ngraphFunction->get_ops()) {
                auto outputLayer = outputInfo.find(op->get_friendly_name());
                if (outputLayer != outputInfo.end()) {
                    auto regionYolo = std::dynamic_pointer_cast<ngraph::op::RegionYolo>(op);
                    if (!regionYolo) {
                        throw std::runtime_error("Invalid output type: " +
                            std::string(regionYolo->get_type_info().name) + ". RegionYolo expected");
                    }
                    this->yoloParams[outputLayer->first] = YoloParams(regionYolo);
                }
            }
        } else if (!labels.empty()) {
            throw std::logic_error("Class labels are not supported with IR version older than 10");
        }
        /*
        if (!labels.empty() && static_cast<int>(labels.size()) != num_classes) {
            if (static_cast<int>(labels.size()) == (num_classes - 1))  // if network assumes default "background" class, having no label
                labels.insert(labels.begin(), "fake");
            else {
                throw std::logic_error("The number of labels is different from numbers of model classes");
            }                
        }
        */
        // -----------------------------------------------------------------------------------------------------

        //---------------------4. Loading model to the plugin -----------------------------------------------------
        std::cout << "Loading model to the plugin" << std::endl;
	this->executable_network = ie.LoadNetwork(cnnNetwork, "MYRIAD", {}); 
		// ---------------------5. Create infer request ------------------------------------------------------------
        std::cout << "Create infer request" << std::endl;
	this->async_infer_request_next = executable_network.CreateInferRequestPtr();
	this->async_infer_request_curr = executable_network.CreateInferRequestPtr();
}

void NcsClassifier::predict_async(int imgIdx, Mat image){
	// --------------------------- 6. Do inference ---------------------------------------------------------
	//std::cout << "Processing image id: " << imgIdx << std::endl;
	FrameToBlob(image, this->async_infer_request_curr, this->inputName);

	//this->async_infer_request_next->StartAsync();
	this->async_infer_request_curr->StartAsync();
    //this->async_infer_request_curr->Wait(IInferRequest::WaitMode::RESULT_READY);
	
	if (OK == this->async_infer_request_curr->Wait(IInferRequest::WaitMode::RESULT_READY)) {
		// ---------------------------Process output blobs--------------------------------------------------
        // Processing results of the CURRENT request
        for (auto &output : this->outputInfo) {
            auto output_name = output.first;
            Blob::Ptr blob = this->async_infer_request_curr->GetBlob(output_name);
            ParseYOLOV3Output(yoloParams[output_name], output_name, blob, resized_im_h, resized_im_w, image.rows, image.cols, imgIdx, this->results);
        }
	}
    (*this->num_frame_processed) ++;
	// Final point:
    // in the truly Async mode we swap the NEXT and CURRENT requests for the next iteration
	// this->async_infer_request_curr.swap(this->async_infer_request_next);
}

void inference_job_async(std::queue<std::pair<int, Mat> > & job_queue, NcsClassifier ncs_classifer){
    std::cout << "Starting Inference, Job Queue: " << job_queue.size() << std::endl;
    int xidx = 0;
    Mat xfile;
    while (!job_queue.empty()){
        xidx = job_queue.front().first;
        xfile = job_queue.front().second;
        //std::cout << "Inference on " << job_queue.front().first << std::endl;
        job_queue.pop();
        ncs_classifer.predict_async(xidx, xfile);
    }
    // One extra images;
    //ncs_classifer.predict_async(xidx+1, xfile);
    std::cout << "In inference_job_async " << ncs_classifer.results->size() << std::endl;
}

class Scheduler{
	public:
	    std::queue<std::pair<int, Mat> > queue;
        std::list<NcsClassifier> workers;
	    Scheduler(int deviceids[], int size, std::string model_xml);
     	void init_workers(int ids[], int size, std::string model_xml);
   	    void start(std::list <std::string> xfilelist, std::string outfilepath);
};

Scheduler::Scheduler(int deviceids[], int size, std::string model_xml){
    init_workers(deviceids, size, model_xml);
}

void Scheduler::init_workers(int ids[], int size, std::string model_xml){
    for (int id=0; id<=size-1; id++){
        std::cout << "creating NcsClassifier for Device " << id << std::endl;
        this->workers.push_back(NcsClassifier(ids[id], this->queue, model_xml));
    }
}

void Scheduler::start(std::list <std::string> xfilelist, std::string outfilepath){
    // start the workers
    std::vector<std::thread> threads;

    //add producer thread for image pre-processing
    int i = 0;
    for (std::string mfile : xfilelist){
        Mat image = imread(mfile);
        this->queue.push({i, image});
        i++;
    }

    std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
    // schedule workers
    for (auto const& worker : this->workers){
	    threads.push_back(std::thread(inference_job_async, std::ref(this->queue), worker));
    }

    // wait for all workers finish
    for (std::thread & th : threads){
	    th.join();
    }

    for (auto const& worker : this->workers) {
        std::cout << "Just after join " << worker.results->size() << std::endl;
    }

    std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time);
    auto ms = milliseconds.count();
    std::cout << "Processing time in ms: " << ms << std::endl;

    // Save results. TODO: make it better
    std::ofstream outfile;
    outfile.open(outfilepath);
    //std::cout << this->workers.size() << std::endl;
    for (auto const& worker : this->workers) {
        std::cout << "Worker ID: " << worker.id << *worker.num_frame_processed << std::endl;
        for (auto const & p : *worker.results) {
            int imgIdx = p.first;
            DetectionObject d = p.second;
            //std::cout << imgIdx << "," << d.xmin << "," << d.ymin << "," << d.xmax << "," << d.ymax << "," << d.confidence << std::endl;
            outfile << (imgIdx+1) << "," << d.xmin << "," << d.ymin << "," << d.xmax << "," << d.ymax << "," << d.confidence << std::endl;
        }
    }
    outfile.close();
}

void run(std::string img_path,int device_ids[], int size, std::string model_xml, std::string outfilepath){
    // scan all files under img_path
    std::list<std::string> xlist;
    for (const auto& xfile : fs::directory_iterator(img_path)){
        xlist.push_back(xfile.path());
    }
    xlist.sort();
    // for (const auto & xfile: xlist) {
    //         std::cout << xfile << std::endl;
    // }
    //init scheduler
    Scheduler x = Scheduler(device_ids, size, model_xml);

    // start processing and wait for complete
    x.start(xlist, outfilepath);
}
	
void showUsage()
{
    std::cout << std::endl;
    std::cout << "[usage]" << std::endl;
    std::cout << "\tutorial1 [option]" << std::endl;
    std::cout << "\toptions:" << std::endl;
    std::cout << std::endl;
    std::cout << "\t\t-h              " << help_message << std::endl;
    std::cout << "\t\t-imgpath <path>       " << input_message << std::endl;
    std::cout << "\t\t-model <path>   " << model_message << std::endl;
	std::cout << "\t\t-num_device #   " << devices_message << std::endl;
	std::cout << "\t\t-labelpath <path>   " << labels_message << std::endl;
    std::cout << "\t\t-outfilepath <path> " << outfile_message << std::endl;
    std::cout << "\t\t-auto_resize        " << input_resizable_message << std::endl;

}


int main(int argc, char *argv[]) {

	gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);

	if (FLAGS_num_device == 0) {
		showUsage();
		return 1;
	}
	if ((FLAGS_imgpath.empty())||(FLAGS_model.empty())) {
		std::cout << "ERROR: input and model file required" << std::endl;
		showUsage();
		return 1;
	}
    
    int device_ids[FLAGS_num_device];
    for (int i=0; i<FLAGS_num_device; ++i){
        device_ids[i] = i;
    }
    int size = FLAGS_num_device;
    std::cout << "Number Devices:" << std::endl;
    std::cout << size  << std::endl;

    std::cout << "imgpath \n";
    std::cout << FLAGS_imgpath;
    std::cout << "\ndevice ids \n";
    for (int elem : device_ids)
        std::cout << elem << '\n';

    createLabels(FLAGS_labelpath);
    run(FLAGS_imgpath, device_ids, size, FLAGS_model, FLAGS_outfilepath);
}

