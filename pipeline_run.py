"""
This module runs the machine learning pipeline.
"""

from pipeline import MLPipeline, PipelineStage

#### Initialize the pipeline
#pipeline = MLPipeline(data_key='bank_data')

#### Run the entire pipeline
#pipeline.run_pipeline()


#### Run specific stages:
pipeline = MLPipeline(data_key='laptops')
pipeline.run_stage(PipelineStage.DETECT_DATASETS)
pipeline.run_stage(PipelineStage.LOAD_CONFIG)
pipeline.run_stage(PipelineStage.LOAD_DATA)

#### Run a range of stages:
#pipeline.run_pipeline(
#    start_stage=PipelineStage.LOAD_CONFIG,
#    end_stage=PipelineStage.PREPROCESS
#)
