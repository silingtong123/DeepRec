# Group Embedding

## Background
DeepRec has done lots of Op Fusion in single EmbeddingVariable Subgraph(Mostly fuse used-less ops introduced by `tf.nn.embedding_lookup_sparse` ) which outperform the native Tensorflow Runtime.However, in a typical CTR scenario, there will actually be hundreds of EmbeddingLookups. Through profiling, we found that the problem of multiple Op kernels launch brought by EmbeddingLookups on the ps side is the bottleneck. So, we hope to develop a function of simultaneously aggregation of multiple EmbeddingLookup, so as to improve EmbeddingLookup performance of this scene.

The Group Embedding function supports simultaneously aggregated multiple EmbeddingLookups on GPUs, and EmbeddingVariable can be set on different GPUs (the default is to evenly distribute according to the number of GPUs used).
This function is designed to support single-card Fusion and multi-card Fusion. Currently, only multi-card Fusion is provided. All keys will be distributed to the corresponding GPUs through collective communication. After the EmbeddingLookup is completed, the obtained Embedding values ​​will also be broadcast to each worker through collective communication.

## Usage

### Environment configuration
First of all, we need to ensure that SOK module is compiled.
```bash
bazel --output_base /tmp build -j 16  -c opt --config=opt  //tensorflow/tools/pip_package:build_sok && ./bazel-bin/tensorflow/tools/pip_package/build_sok
```

### User API
1. Before using Group Embedding API, you need to enable the setting`tf.config.experimental.enable_group_embedding()`
parameters setting: ```fusion_type="collective"```. This mode will complete the initialization of the horovod module and SOK-related dependencies.

2. We support two levels of API.The one is`tf.nn.group_embedding_lookup_sparse` and the other is `tf.feature_column.group_embedding_column_scope` which is based on feature_column API.

**group_embedding_lookup_sparse**

```python
def group_embedding_lookup_sparse(params,
                                  sp_ids,
                                  combiners,
                                  partition_strategy="mod",
                                  sp_weights=None,
                                  name=None):
```

- `params` : List, This parameter could receive one or more EmbeddingVariables.
- `sp_ids` : List | Tuple , SparseTensor sp_ids ​​is the ID used for EmbeddingLookup, the length must be consistent with params.
- `combiners` : List | Tuple，The pooling method of embedding values.Currently support `mean` and `sum`.
- `partition_strategy` : str，Currently not supported.
- `sp_weights` : List | Typle the weight of sp_ids values.(Currently not supported.)
- `name` : str group name

**group_embedding_column_scope**

```python
def group_embedding_column_scope(name=None):
```
- `name` ： The name of scope.

We need to initialize a context `group_embedding_column_scope`
, and complete the construction of `EmbeddingColumn` in that context. Later, the EmbeddingColumn Lookup would be simultaneously aggregate by `tf.feature_column.input_layer`. It is worth noting that the underlying implementation of this function is designed for `tf.RaggedTensor`。Although the IDS for EmbeddingLookup also supports `SparseTensor`, it will still be converted to `RaggedTensor` in the end, which will introduce a certain performance overhead.

### Example

**Usage of group_embedding_lookup**

```python
import tensorflow as tf

tf.config.experimental.enable_group_embedding(fusion_type="collective")

ev_opt = tf.EmbeddingVariableOption(evict_option=None,
                                    filter_option=None)

with tf.device('/GPU:{}'.format(0)):
    var_0 = tf.get_embedding_variable("var_0",
                                    embedding_dim=16,
                                    initializer=tf.ones_initializer(tf.float32),
                                    ev_option=ev_opt)
    
    var_1 = tf.get_embedding_variable("var_1",
                                    embedding_dim=8,
                                    initializer=tf.ones_initializer(tf.float32),
                                    ev_option=ev_opt)

##We recommend use RaggedTensor representation here.
indices_0 = tf.RaggedTensor.from_row_splits(
    values=[3, 1, 4, 1, 5, 9, 2, 6],
    row_splits=[0, 4, 4, 7, 8, 8])

embedding_weights = [var_0, var_1]
indices = [indices_0 for _ in range(2)]
combiners = ["sum", "sum"]

deep_features = tf.nn.group_embedding_lookup_sparse(embedding_weights, indices, combiners)

init = tf.global_variables_initializer()
sess_config = tf.ConfigProto()
sess_config.gpu_options.visible_device_list = "0" #str(hvd.local_rank())
sess_config.gpu_options.allow_growth = True
with tf.Session(config=sess_config) as sess:
  sess.run(init)
  print("init global done")
  print(sess.run([deep_features]))
```

**Usage of feature_column**

```python
from tensorflow.python.feature_column import feature_column as fc_old
from tensorflow.python.framework import dtypes
tf.config.experimental.enable_group_embedding(fusion_type="collective")

ev_opt = tf.EmbeddingVariableOption(evict_option=None,
                                    filter_option=None)

# group_name represent for the grouped embedding variable
with tf.feature_column.group_embedding_column_scope(name="item")::
  ad0_col = tf.feature_column.categorical_column_with_embedding(
      key='ad0', dtype=dtypes.int64, ev_option=ev_opt)
  ad0_fc = tf.feature_column.embedding_column(
    categorical_column=ad0_col,
    dimension=20,
    initializer=tf.constant_initializer(0.5),
    group_name='item')
  ad1_fc = tf.feature_column.embedding_column(
    tf.feature_column.categorical_column_with_embedding(
      key='ad1', dtype=dtypes.int64, ev_option=ev_opt),
    dimension=30,
    initializer=tf.constant_initializer(0.5))

with tf.feature_column.group_embedding_column_scope(name="user")::
  user0_fc = tf.feature_column.embedding_column(
      tf.feature_column.categorical_column_with_embedding(
        key='user0', dtype=dtypes.int64, ev_option=ev_opt),
      dimension=20,
      initializer=tf.constant_initializer(0.5))

columns = [ad0_fc, ad1_fc, user0_fc]

ids={}
ids["ad0"] = tf.SparseTensor(indices=[[0,0],[0,1],[1,1],[2,2],[3,3],[4,3],[4,4]], \
                        values=tf.cast([1,3,2,3,4,5,3], tf.dtypes.int64), dense_shape=[5, 5])    
ids["ad1"] = tf.SparseTensor(indices=[[0,0],[0,1],[1,1],[2,2],[3,3],[4,3],[4,4]], \
                        values=tf.cast([1,3,2,3,4,5,3], tf.dtypes.int64), dense_shape=[5, 5])
ids["user0"] = tf.SparseTensor(indices=[[0,0],[0,1],[1,1],[2,2],[3,3],[4,3],[4,4]], \
                        values=tf.cast([1,3,2,3,4,5,3], tf.dtypes.int64), dense_shape=[5, 5])   

emb = tf.feature_column.input_layer(ids, columns)
fun = tf.multiply(emb, 2.0, name='multiply')
loss = tf.reduce_sum(fun, name='reduce_sum')
opt = tf.train.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
g_v = opt.compute_gradients(loss)
train_op = opt.apply_gradients(g_v)
init = tf.global_variables_initializer()

sess_config = tf.ConfigProto()
sess_config.gpu_options.visible_device_list = "0" #(hvd.local_rank())
sess_config.gpu_options.allow_growth = True
with tf.Session(config=sess_config) as sess:
    sess.run(init)
    print("init global done")
    print(sess.run([fun, train_op,loss]))
```

More detailed usage of group embedding lookup please refer to modelzoo[DCNv2模型示例](../../modelzoo/features/group_embedding_lookup/dcnv2/train.py)

**Benchmarks**

We evaluate performance with DCNv2 model. The training dataset is Criteo and the batch_size is 512.

| API | Global-step/s | Note |
| ------ | ---------------------- | ---- |
| tf.nn.group_embedding_lookup_sparse            | 85.3 (+/- 0.1)  | 1 card |
| tf.nn.group_embedding_lookup_sparse            | 122.2 (+/- 0.2) | 2 card |
| tf.nn.group_embedding_lookup_sparse            | 212.4 (+/- 0.4) | 4 card |
| tf.feature_column.group_embedding_column_scope | 79.7 (+/- 0.1)  | 1 card |
| tf.feature_column.group_embedding_column_scope | 152.2 (+/- 0.2) | 2 card |
| tf.feature_column.group_embedding_column_scope | 272 (+/- 0.4)   | 4 card |