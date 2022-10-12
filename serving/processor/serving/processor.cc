#include "processor.h"
#include "model_serving.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "serving/processor/serving/predict.pb.h"
#include <future> 
#include <thread>

extern "C" {

void* initializeImpl(const char* model_entry, const char* model_config,
                 int* state) {
  auto model = new tensorflow::processor::Model(std::string(model_entry));
  auto status = model->Init(model_config);
  if (!status.ok()) {
    std::cerr << "[TensorFlow] Processor initialize failed"
              << ", status:" << status.error_message() << std::endl;
    *state = -1;
    return nullptr;
  }
  
  std::cout << "[TensorFlow] Processor initialize success." << std::endl;

  *state = 0;
  return model;
}

void* initialize(const char* model_entry, const char* model_config,
                 int* state) {
      std::packaged_task<void *(const char*, const char*, int*)> task(initializeImpl);
      std::future<void *> future = task.get_future();
      std::thread t(std::move(task), model_entry, model_config, state);
      t.detach();
      return future.get();     
}

int updateImpl(void* model_buf){
  if (model_buf == nullptr) {
     return -1;
  }
  auto model = static_cast<tensorflow::processor::Model*>(model_buf);
  return model->Update();
}

int update(void* model_buf) {
      std::packaged_task<int(void *)> task(updateImpl);
      std::future<int> future = task.get_future();
      std::thread t(std::move(task), model_buf);
      t.detach();
      return future.get();   
}

int processImpl(void* model_buf, const void* input_data, int input_size,
            void** output_data, int* output_size) {
  auto model = static_cast<tensorflow::processor::Model*>(model_buf);
  if (input_size == 0) {
    auto model_str = model->DebugString();
    *output_data = strndup(model_str.c_str(), model_str.length());
    *output_size = model_str.length();
    return 200;
  }
  auto status = model->Predict(input_data, input_size,
      output_data, output_size);
  if (!status.ok()) {
    std::string errmsg = tensorflow::strings::StrCat(
        "[TensorFlow] Processor predict failed: ",
        status.error_message());
    *output_data = strndup(errmsg.c_str(), strlen(errmsg.c_str()));
    *output_size = strlen(errmsg.c_str());
    LOG(ERROR) << errmsg;
    return 500;
  }
  return 200;
}

int process(void* model_buf, const void* input_data, int input_size,
            void** output_data, int* output_size) {
      std::packaged_task<int(void *, const void*, int , void **, int *)> task(processImpl);
      std::future<int> future = task.get_future();
      std::thread t(std::move(task), model_buf, input_data, input_size, output_data, output_size);
      t.detach();
      return future.get();     
}

int batch_process(void* model_buf, const void* input_data[], int* input_size,
                  void* output_data[], int* output_size) {
  auto model = static_cast<tensorflow::processor::Model*>(model_buf);
  if (input_size == 0) {
    auto model_str = model->DebugString();
    *output_data = strndup(model_str.c_str(), model_str.length());
    *output_size = model_str.length();
    return 200;
  }

  auto status = model->BatchPredict(input_data, input_size,
      output_data, output_size);
  if (!status.ok()) {
    std::string errmsg = tensorflow::strings::StrCat(
        "[TensorFlow] Processor predict failed: ",
        status.error_message());
    *output_data = strndup(errmsg.c_str(), strlen(errmsg.c_str()));
    *output_size = strlen(errmsg.c_str());
    LOG(ERROR) << errmsg;
    return 500;
  }
  return 200;
}

// TODO: EAS has a higher priority to call async interface.
//       Now we have no implementation of this async interface,
//       this will block EAS to call sync interface.
//
//typedef void (*DoneCallback)(const char* output, int output_size,
//    int64_t request_id, int finished, int error_code);
//
//void async_process(void* model_buf, const void* input_data,
//    int input_size, int64_t request_id, DoneCallback done) {
//  // TODO
//}

int get_serving_model_info(
    void* model_buf, void** output_data, int* output_size) {
  auto model = static_cast<tensorflow::processor::Model*>(model_buf);
  auto status = model->GetServingModelInfo(output_data, output_size);
  if (!status.ok()) {
    std::string errmsg = tensorflow::strings::StrCat(
        "[TensorFlow] Processor get serving model info failed: ",
        status.error_message());
    *output_data = strndup(errmsg.c_str(), strlen(errmsg.c_str()));
    *output_size = strlen(errmsg.c_str());
    LOG(ERROR) << errmsg;
    return 500;
  }
  return 200;
}

} // extern "C"
