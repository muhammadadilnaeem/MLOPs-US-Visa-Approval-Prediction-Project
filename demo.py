

# # demo to check if logging is working 
# from us_visa_project.logger import logging

# logging.info("Hello World! Let's Start the project")


## demo to check if our custom excemption class is working

# import sys
# from us_visa_project.exception import USVISAException
# from us_visa_project.logger import logging

# logging.info("We got the error in our code, let's handle the project")

# try:
#     a = 1/0
# except Exception as e:
#     raise USVISAException(e,sys)


# # let's check if MONGODB_URL_KEY is set or not
# from us_visa_project.constants import MONGODB_URL_KEY
# print(MONGODB_URL_KEY)

# import os
# mongo_db_url = os.getenv("MONGODB_URL")
# print(mongo_db_url)


from us_visa_project.pipline.training_pipeline import TrainPipeline

obj = TrainPipeline()
obj.run_pipeline()
