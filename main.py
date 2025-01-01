from src.datascience import logger
from src.datascience.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.datascience.pipeline.data_validation_pipeline import DataValidationTrainingPipeline
from src.datascience.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
from src.datascience.pipeline.model_trainer_pipeline import ModelTrainerTrainingPipeline
from src.datascience.pipeline.model_evaluation_pipeline import ModelEvaluationPipeline

STAGE_NAME="DATA INGESTION STAGE"
try:
    logger.info(f">>>>>>stage {STAGE_NAME} started<<<<<<")
    obj=DataIngestionTrainingPipeline()
    obj.initiate_data_ingestion()
    logger.info(f">>>>>stage {STAGE_NAME} completed <<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME="DATA VALIDATION STAGE"
try:
    logger.info(f">>>>>>stage {STAGE_NAME} started<<<<<<")
    obj=DataValidationTrainingPipeline()
    obj.initiate_data_validation()
    logger.info(f">>>>>stage {STAGE_NAME} completed <<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME="DATA TRANSFORMATION STAGE"
try:
    logger.info(f">>>>>>stage {STAGE_NAME} started<<<<<<")
    obj=DataTransformationTrainingPipeline()
    obj.initiate_data_transformation()
    logger.info(f">>>>>stage {STAGE_NAME} completed <<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME="MODEL TRAINING STAGE"
try:
    logger.info(f">>>>>>stage {STAGE_NAME} started<<<<<<")
    obj=ModelTrainerTrainingPipeline()
    obj.initiate_model_trainer()
    logger.info(f">>>>>stage {STAGE_NAME} completed <<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME="MODEL EVALUATION STAGE"
try:
    logger.info(f">>>>>>stage {STAGE_NAME} started<<<<<<")
    obj=ModelEvaluationPipeline()
    obj.initiate_model_evaluation()
    logger.info(f">>>>>stage {STAGE_NAME} completed <<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e