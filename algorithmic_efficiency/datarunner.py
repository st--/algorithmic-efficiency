from algorithmic_efficiency import workloads.criteo1tb.input_pipeline as input_pipeline
import time
import jax


ds = input_pipeline.get_criteo1tb_dataset(split='train', 
                                        shuffle_rng=jax.random.PRNGKey(0), 
                                        data_dir='/home/kasimbeg/data/criteo1tb'
                                        num_dense_features=13,
                                        global_batch_size=262_144,
)

for i in 500:
    time.sleep(2)
    next_batch = next(ds)